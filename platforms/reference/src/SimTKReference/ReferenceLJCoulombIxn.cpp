
/* Portions copyright (c) 2006-2018 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "SimTKOpenMMUtilities.h"
#include "ReferenceLJCoulombIxn.h"
#include "ReferenceForce.h"
#include "ReferencePME.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::set;
using std::vector;
using namespace OpenMM;

typedef int    ivec[3];

/**---------------------------------------------------------------------------------------

   ReferenceLJCoulombIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceLJCoulombIxn::ReferenceLJCoulombIxn() : cutoff(false), useSwitch(false), periodic(false), ewald(false), pme(false), ljpme(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceLJCoulombIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceLJCoulombIxn::~ReferenceLJCoulombIxn() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use
     @param solventDielectric   the dielectric constant of the bulk solvent

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors, double solventDielectric) {

    cutoff = true;
    cutoffDistance = distance;
    neighborList = &neighbors;
    krf = pow(cutoffDistance, -3.0)*(solventDielectric-1.0)/(2.0*solventDielectric+1.0);
    crf = (1.0/cutoffDistance)*(3.0*solventDielectric)/(2.0*solventDielectric+1.0);
}

/**---------------------------------------------------------------------------------------

   Set the force to use a switching function on the Lennard-Jones interaction.

   @param distance            the switching distance

   --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUseSwitchingFunction(double distance) {
    useSwitch = true;
    switchingDistance = distance;
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setPeriodic(OpenMM::Vec3* vectors) {

    assert(cutoff);
    assert(vectors[0][0] >= 2.0*cutoffDistance);
    assert(vectors[1][1] >= 2.0*cutoffDistance);
    assert(vectors[2][2] >= 2.0*cutoffDistance);
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

     Set the force to use Ewald summation.

     @param alpha  the Ewald separation parameter
     @param kmaxx  the largest wave vector in the x direction
     @param kmaxy  the largest wave vector in the y direction
     @param kmaxz  the largest wave vector in the z direction

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz) {
    alphaEwald = alpha;
    numRx = kmaxx;
    numRy = kmaxy;
    numRz = kmaxz;
    ewald = true;
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation.

     @param alpha  the Ewald separation parameter
     @param gridSize the dimensions of the mesh

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUsePME(double alpha, int meshSize[3]) {
    alphaEwald = alpha;
    meshDim[0] = meshSize[0];
    meshDim[1] = meshSize[1];
    meshDim[2] = meshSize[2];
    pme = true;
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation for dispersion terms.

     @param alpha  the dispersion Ewald separation parameter
     @param gridSize the dimensions of the dispersion mesh

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUseLJPME(double alpha, int meshSize[3]) {
    alphaDispersionEwald = alpha;
    dispersionMeshDim[0] = meshSize[0];
    dispersionMeshDim[1] = meshSize[1];
    dispersionMeshDim[2] = meshSize[2];
    ljpme = true;
}

/**---------------------------------------------------------------------------------------

   Calculate Ewald ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                              vector<vector<double> >& atomParameters, vector<set<int> >& exclusions,
                                              vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal, 
                                              double* vext_grid, const vector<int>& QMexclude, vector<Vec3>& PME_grid_positions ) const {
    typedef std::complex<double> d_complex;

    static const double epsilon     =  1.0;

    int kmax                            = (ewald ? std::max(numRx, std::max(numRy,numRz)) : 0);
    double factorEwald              = -1 / (4*alphaEwald*alphaEwald);
    double SQRT_PI                  = sqrt(PI_M);
    double TWO_PI                   = 2.0 * PI_M;
    double recipCoeff               = ONE_4PI_EPS0*4*PI_M/(periodicBoxVectors[0][0] * periodicBoxVectors[1][1] * periodicBoxVectors[2][2]) /epsilon;

    double totalSelfEwaldEnergy     = 0.0;
    double realSpaceEwaldEnergy     = 0.0;
    double recipEnergy              = 0.0;
    double recipDispersionEnergy    = 0.0;
    double totalRecipEnergy         = 0.0;
    double vdwEnergy                = 0.0;

    // data structures for computing vext_grid, make sure to free these at return!
    std::vector<int> ngrid;    // number of PME grid points
    ivec* particleindex;       // atom grid indices
    // I'd rather not keep checking for null pointer, so let's introduce boolean
    bool compute_vext_grid=false;
    if(vext_grid){ compute_vext_grid=true; }


    // A couple of sanity checks for
    if(ljpme && useSwitch)
        throw OpenMMException("Switching cannot be used with LJPME");
    if(ljpme && !pme)
        throw OpenMMException("LJPME has been set, without PME being set");

    /*
    printf("in ReferenceLJCoulombIxn\n");

    for(int i=0; i < QMexclude.size(); i++){
        printf(" exclusion %d  %d \n" , i , QMexclude[i] );
    }
    */

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {
        for (int atomID = 0; atomID < numberOfAtoms; atomID++) {
            double selfEwaldEnergy       = ONE_4PI_EPS0*atomParameters[atomID][QIndex]*atomParameters[atomID][QIndex] * alphaEwald/SQRT_PI;
            if(ljpme) {
                // Dispersion self term
                selfEwaldEnergy -= pow(alphaDispersionEwald, 6.0) * 64.0*pow(atomParameters[atomID][SigIndex], 6.0) * pow(atomParameters[atomID][EpsIndex], 2.0) / 12.0;
            }
            totalSelfEwaldEnergy            -= selfEwaldEnergy;
        }
    }

    if (totalEnergy) {
        *totalEnergy += totalSelfEwaldEnergy;
    }

    // **************************************************************************************
    // RECIPROCAL SPACE EWALD ENERGY AND FORCES
    // **************************************************************************************
    // PME

    if (pme && includeReciprocal) {
        pme_t          pmedata; /* abstract handle for PME data */

        // if computing vext grid, call overloaded pme_init for this setup
        if(compute_vext_grid)
            { pme_init(&pmedata,alphaEwald,numberOfAtoms,meshDim,5,1,compute_vext_grid);}
        else
            { pme_init(&pmedata,alphaEwald,numberOfAtoms,meshDim,5,1);}

        vector<double> charges(numberOfAtoms);
        for (int i = 0; i < numberOfAtoms; i++)
            charges[i] = atomParameters[i][QIndex];
        pme_exec(pmedata,atomCoordinates,forces,charges,periodicBoxVectors,&recipEnergy);

        if (totalEnergy)
            *totalEnergy += recipEnergy;

        // Computing Vext grid.  We need to copy over several pme data structures and save
        // locally before call to pme_destroy(pmedata)
        if (compute_vext_grid)
        {
            // grid size
            ngrid = pme_return_gridsize(pmedata);
            int nx = ngrid[0];
            int ny = ngrid[1];
            int nz = ngrid[2];

            // copy particle grid indices
            particleindex = (ivec *) malloc(sizeof(ivec)*numberOfAtoms);
            pme_copy_particleindex( pmedata, particleindex );

            // save electrostatic potential on PME grid, before pme_destroy
            // copy pme grid over to vext_grid.  After call of pme_exec, pme grid should by default store the external potential
            pme_copy_grid_real( pmedata, vext_grid );
        }


        pme_destroy(pmedata);

        if (ljpme) {
            // Dispersion reciprocal space terms
            pme_init(&pmedata,alphaDispersionEwald,numberOfAtoms,dispersionMeshDim,5,1);

            std::vector<Vec3> dpmeforces(numberOfAtoms);
            for (int i = 0; i < numberOfAtoms; i++)
                charges[i] = 8.0*pow(atomParameters[i][SigIndex], 3.0) * atomParameters[i][EpsIndex];
            pme_exec_dpme(pmedata,atomCoordinates,dpmeforces,charges,periodicBoxVectors,&recipDispersionEnergy);
            for (int i = 0; i < numberOfAtoms; i++)
                forces[i] += dpmeforces[i];
            if (totalEnergy)
                *totalEnergy += recipDispersionEnergy;
            pme_destroy(pmedata);
        }
    }
    // Ewald method

    else if (ewald && includeReciprocal) {

        // setup reciprocal box

        double recipBoxSize[3] = { TWO_PI / periodicBoxVectors[0][0], TWO_PI / periodicBoxVectors[1][1], TWO_PI / periodicBoxVectors[2][2]};


        // setup K-vectors

#define EIR(x, y, z) eir[(x)*numberOfAtoms*3+(y)*3+z]
        vector<d_complex> eir(kmax*numberOfAtoms*3);
        vector<d_complex> tab_xy(numberOfAtoms);
        vector<d_complex> tab_qxyz(numberOfAtoms);

        if (kmax < 1)
            throw OpenMMException("kmax for Ewald summation < 1");

        for (int i = 0; (i < numberOfAtoms); i++) {
            for (int m = 0; (m < 3); m++)
                EIR(0, i, m) = d_complex(1,0);

            for (int m=0; (m<3); m++)
                EIR(1, i, m) = d_complex(cos(atomCoordinates[i][m]*recipBoxSize[m]),
                                         sin(atomCoordinates[i][m]*recipBoxSize[m]));

            for (int j=2; (j<kmax); j++)
                for (int m=0; (m<3); m++)
                    EIR(j, i, m) = EIR(j-1, i, m) * EIR(1, i, m);
        }

        // calculate reciprocal space energy and forces

        int lowry = 0;
        int lowrz = 1;

        for (int rx = 0; rx < numRx; rx++) {

            double kx = rx * recipBoxSize[0];

            for (int ry = lowry; ry < numRy; ry++) {

                double ky = ry * recipBoxSize[1];

                if (ry >= 0) {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n] = EIR(rx, n, 0) * EIR(ry, n, 1);
                }

                else {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n]= EIR(rx, n, 0) * conj (EIR(-ry, n, 1));
                }

                for (int rz = lowrz; rz < numRz; rz++) {

                    if (rz >= 0) {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_qxyz[n] = atomParameters[n][QIndex] * (tab_xy[n] * EIR(rz, n, 2));
                    }

                    else {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_qxyz[n] = atomParameters[n][QIndex] * (tab_xy[n] * conj(EIR(-rz, n, 2)));
                    }

                    double cs = 0.0f;
                    double ss = 0.0f;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        cs += tab_qxyz[n].real();
                        ss += tab_qxyz[n].imag();
                    }

                    double kz = rz * recipBoxSize[2];
                    double k2 = kx * kx + ky * ky + kz * kz;
                    double ak = exp(k2*factorEwald) / k2;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        double force = ak * (cs * tab_qxyz[n].imag() - ss * tab_qxyz[n].real());
                        forces[n][0] += 2 * recipCoeff * force * kx ;
                        forces[n][1] += 2 * recipCoeff * force * ky ;
                        forces[n][2] += 2 * recipCoeff * force * kz ;
                    }

                    recipEnergy       = recipCoeff * ak * (cs * cs + ss * ss);
                    totalRecipEnergy += recipEnergy;

                    if (totalEnergy)
                        *totalEnergy += recipEnergy;

                    lowrz = 1 - numRz;
                }
                lowry = 1 - numRy;
            }
        }
    }

    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (!includeDirect)
        return;
    double totalVdwEnergy            = 0.0f;
    double totalRealSpaceEwaldEnergy = 0.0f;


    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r         = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
        double switchValue = 1, switchDeriv = 0;
        if (useSwitch && r > switchingDistance) {
            double t = (r-switchingDistance)/(cutoffDistance-switchingDistance);
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
            switchDeriv = t*t*(-30+t*(60-t*30))/(cutoffDistance-switchingDistance);
        }
        double alphaR = alphaEwald * r;


        double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
        dEdR = dEdR * (erfc(alphaR) + 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

        double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
        double sig2 = inverseR*sig;
        sig2 *= sig2;
        double sig6 = sig2*sig2*sig2;
        double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
        dEdR += switchValue*eps*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
        vdwEnergy = eps*(sig6-1.0)*sig6;

        if (ljpme) {
            double dalphaR   = alphaDispersionEwald * r;
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double dar6 = dar4*dar2;
            double inverseR2 = inverseR*inverseR;
            double c6i = 8.0*pow(atomParameters[ii][SigIndex], 3.0) * atomParameters[ii][EpsIndex];
            double c6j = 8.0*pow(atomParameters[jj][SigIndex], 3.0) * atomParameters[jj][EpsIndex];
            // For the energies and forces, we first add the regular Lorentz−Berthelot terms.  The C12 term is treated as usual
            // but we then subtract out (remembering that the C6 term is negative) the multiplicative C6 term that has been
            // computed in real space.  Finally, we add a potential shift term to account for the difference between the LB
            // and multiplicative functional forms at the cutoff.
            double emult = c6i*c6j*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2) * (1.0 + dar2 + 0.5*dar4));
            dEdR += 6.0*c6i*c6j*inverseR2*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2) * (1.0 + dar2 + 0.5*dar4 + dar6/6.0));

            double inverseCut2 = 1.0/(cutoffDistance*cutoffDistance);
            double inverseCut6 = inverseCut2*inverseCut2*inverseCut2;
            sig2 = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
            sig2 *= sig2;
            sig6 = sig2*sig2*sig2;
            // The additive part of the potential shift
            double potentialshift = eps*(1.0-sig6*inverseCut6)*sig6*inverseCut6;
            dalphaR   = alphaDispersionEwald * cutoffDistance;
            dar2 = dalphaR*dalphaR;
            dar4 = dar2*dar2;
            // The multiplicative part of the potential shift
            potentialshift -= c6i*c6j*inverseCut6*(1.0 - EXP(-dar2) * (1.0 + dar2 + 0.5*dar4));
            vdwEnergy += emult + potentialshift;
        }

        if (useSwitch) {
            dEdR -= vdwEnergy*switchDeriv*inverseR;
            vdwEnergy *= switchValue;
        }

        // accumulate forces

        for (int kk = 0; kk < 3; kk++) {
            double force  = dEdR*deltaR[0][kk];
            forces[ii][kk]   += force;
            forces[jj][kk]   -= force;
        }

        // accumulate energies

        realSpaceEwaldEnergy        = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erfc(alphaR);

        totalVdwEnergy             += vdwEnergy;
        totalRealSpaceEwaldEnergy  += realSpaceEwaldEnergy;

    }

    if (totalEnergy)
        *totalEnergy += totalRealSpaceEwaldEnergy + totalVdwEnergy;

    // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

    double totalExclusionEnergy = 0.0f;
    const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
    for (int i = 0; i < numberOfAtoms; i++)
        for (int exclusion : exclusions[i]) {
            if (exclusion > i) {
                int ii = i;
                int jj = exclusion;

                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                double alphaR    = alphaEwald * r;
                if (erf(alphaR) > 1e-6) {
                    double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
                    dEdR = dEdR * (erf(alphaR) - 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

                    // accumulate forces

                    for (int kk = 0; kk < 3; kk++) {
                        double force = dEdR*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }

                    // accumulate energies

                    realSpaceEwaldEnergy = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erf(alphaR);
                }
                else {
                    realSpaceEwaldEnergy = alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex];
                }

                if(ljpme){
                    // Dispersion terms.  Here we just back out the reciprocal space terms, and don't add any extra real space terms.
                    double dalphaR   = alphaDispersionEwald * r;
                    double inverseR2 = inverseR*inverseR;
                    double dar2 = dalphaR*dalphaR;
                    double dar4 = dar2*dar2;
                    double dar6 = dar4*dar2;
                    double c6i = 8.0*pow(atomParameters[ii][SigIndex], 3.0) * atomParameters[ii][EpsIndex];
                    double c6j = 8.0*pow(atomParameters[jj][SigIndex], 3.0) * atomParameters[jj][EpsIndex];
                    realSpaceEwaldEnergy -= c6i*c6j*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2) * (1.0 + dar2 + 0.5*dar4));
                    double dEdR = -6.0*c6i*c6j*inverseR2*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2) * (1.0 + dar2 + 0.5*dar4 + dar6/6.0));
                    for (int kk = 0; kk < 3; kk++) {
                        double force = dEdR*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }
                }

                totalExclusionEnergy += realSpaceEwaldEnergy;
            }
        }

    if (totalEnergy)
        *totalEnergy -= totalExclusionEnergy;


    if (!compute_vext_grid)
        return;


    /* Computing Electrostatic Potential on PME grid, already have reciprocal space, now do real space ... */
    // pme grid stores electrostatic potential after calls to
    //     pme_reciprocal_convolution(pme,periodicBoxVectors,recipBoxVectors,energy);
    //     fftpack_exec_3d(pme->fftplan,FFTPACK_BACKWARD,pme->grid,pme->grid);

    int nx = ngrid[0];
    int ny = ngrid[1];
    int nz = ngrid[2];

  
    // get reciprocal box vectors
    Vec3 recipBoxVectors[3];
    invert_box_vectors(periodicBoxVectors , recipBoxVectors);
    // maximum grid displacements corresponding to realspace cutoff distance
    int dgrid[3];
    for (int i =0; i < 3; i++){
        double recip_norm = sqrt( recipBoxVectors[0][i]*recipBoxVectors[0][i] + recipBoxVectors[1][i]*recipBoxVectors[1][i] + recipBoxVectors[2][i]*recipBoxVectors[2][i]);
        dgrid[i] = floor( cutoffDistance*recip_norm*ngrid[i] ) + 1;
    }


    // Use neighbor list to find atoms within cutoff from QM region.  Fill a vector with all neighbors
    // of every atom in the QMregion, and use these for real-space interactions with the PMEgrid

    printf(" searching for QM neighbors from system neighbor list ...");

    std::unordered_set<int> QMneighbors;  // data structure which we will fill with neighbors of QM region.
//****** first add QMexclude atoms to QMneighbors
    for(int i=0; i < QMexclude.size(); i++) {
        int index = QMexclude[i];
        QMneighbors.insert(index);
    }

// *******    This collects neighbors from entire QM region.  Note there's a lot of vector searching,
// *******    so will be slow---if this is a problem, could force QMexclude to be ordered, then use binary search...
    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;
        if ( std::find(QMexclude.begin(), QMexclude.end(), ii) != QMexclude.end() ) {
            // this is QMregion atom, insert (only if not in set)
             QMneighbors.insert(jj);
        }
        else if ( std::find(QMexclude.begin(), QMexclude.end(), jj) != QMexclude.end() ) {
            // this is QMregion atom, insert (only if not in set)
            QMneighbors.insert(ii);
        }
    }

    printf(" done searching for QM neighbors");

    // Keep track of exclusions from QM region.  Don't include contribution to vext grid from QM atoms,
    // need to subtract off these contributions from reciprocal space
    // not the cleanest way to do it, but we introduce new data structure with zero's and one's ...

    int* QMexclusion_flag;       // QM_exclusions
    QMexclusion_flag = (int *) malloc(sizeof(int)*numberOfAtoms);
    // initialize data structure with zero's...
    for (int i =0; (i < numberOfAtoms); i++)
        QMexclusion_flag[i] = 0;
    // set to 1's for QM atoms
    for(int i=0; i < QMexclude.size(); i++) {
        int index = QMexclude[i];
        QMexclusion_flag[index]=1;
    }

    /*   
    printf(" ********** QM exclusion map ************** \n");
    for (int i =0; (i < numberOfAtoms); i++)
        printf( "%d  %d \n" , i, QMexclusion_flag[i] );
    */

    // Store absolute positions of PME grid cells
    for (int ia = 0; ia < ngrid[0] ; ia++){
        for (int ib = 0; ib < ngrid[1] ; ib++){
            for (int ic = 0; ic < ngrid[2] ; ic++){
                int index = ia*ngrid[1]*ngrid[2] + ib*ngrid[2] + ic;
                PME_grid_positions[index][0]=0.0;
                PME_grid_positions[index][1]=0.0;
                PME_grid_positions[index][2]=0.0;
                int igrid[3];
                igrid[0] = ia ;
                igrid[1] = ib ;
                igrid[2] = ic ;
                for(int j=0; j<3; j++){
                    for(int k=0; k<3; k++)
                        PME_grid_positions[index][k] += (double)igrid[j] / ngrid[j] * periodicBoxVectors[j][k] ; 
                }
            }
        }
    }

 
    //printf( " QM neighbors \n");
    // compute real-space contribution to vext grid, and add to previously stored reciprocal space contribution.
    //for (int i = 0; (i < numberOfAtoms); i++) {
    for (auto& i : QMneighbors) {
        //printf(" %d  \n" , i );
        double q_i = atomParameters[i][QIndex];
        Vec3 r_i = atomCoordinates[i];

        // nearest PME grid point of atom
        int ix = particleindex[i][0];
        int iy = particleindex[i][1];
        int iz = particleindex[i][2];

        //printf(" PME particle index %d %d %d %d \n" , i , ix , iy , iz );

        int igrid[3];
        // fill in contribution to vext to all grid points within realspace cutoff of this atom
        for (int ia = -dgrid[0]; ia < dgrid[0] + 1; ia++){
            for (int ib = -dgrid[1]; ib < dgrid[1] + 1; ib++){
                for (int ic = -dgrid[2]; ic < dgrid[2] + 1; ic++){

                    // use trick in ReferencePME.cpp to avoid conditionals on PBC search ...
                    igrid[0] = ( ix + ia + ngrid[0] ) % ngrid[0] ;
                    igrid[1] = ( iy + ib + ngrid[1] ) % ngrid[1] ;
                    igrid[2] = ( iz + ic + ngrid[2] ) % ngrid[2] ;

                    int index = igrid[0]*ngrid[1]*ngrid[2] + igrid[1]*ngrid[2] + igrid[2];

                    // get real space distance from atom to this grid point
                    Vec3 r_grid;
                    for(int k=0; k<3; k++)
                        r_grid[k] = PME_grid_positions[index][k];
                    
                    // minimum image, see comments above about orthrhombic box limitation
                    Vec3 dr = ReferenceForce::getDeltaRPeriodic( r_grid , r_i , periodicBoxVectors );
                    double inverseR = 1.0 / sqrt(dr.dot(dr));
                    double alphaR = alphaEwald / inverseR ;  // just alpha * r 

                    // extremely rare to have a divide by zero if particle is exactly on PME grid.
                    // this will almost never happen, so hopefully following conditional is harmless in terms
                    // of speed/slowdown, but better throw an exception rather than not do anything...
                    if ( alphaR < 1e-6 )
                        throw OpenMMException("particle is exactly on PMEgrid, vext will diverge!");


                    /************************* add contribution to vext   **************************************/
                    /*  For enhanced speed, uncomment code in 'if' statement below, and comment entire code block starting with
                    *   "For QMatoms, subtract off ...."  .  The code in the block below subtracts off QM contributions
                    *   to grid points within the real-space cutoff, while the code block "For QMatoms, subtract off ...." subtracts off
                    *   QM contributions to all grid points.  While we probably only need to subtract off QMatom contribution to QMregion,
                    *   and can use the faster code block, we keep the less efficient code as it is easier to have QM contributions removed
                    *   from all PME grid points for testing purposes.
                    */

                    // If QM exclusion, subtract reciprocal space
                    if ( QMexclusion_flag[i] == 1 ){
                        /*
                        if (erf(alphaR) > 1e-6) {
                            vext_grid[index] -= ONE_4PI_EPS0*q_i*inverseR*erf(alphaR); 
                        }
                        else{
                            vext_grid[index] -= alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*q_i;
                        }
                        */
                    }
                    // Not QM exclusion, add real space, subtract recip space
                    else{
                        vext_grid[index] += ONE_4PI_EPS0*q_i*inverseR*erfc(alphaR);
                    }

                }
            }
        }


        //**************** For QMatoms, subtract off contribution for all grid points ******
        if ( QMexclusion_flag[i] == 1 ){

            for (int ia = 0; ia < ngrid[0] ; ia++){
                for (int ib = 0; ib < ngrid[1] ; ib++){
                    for (int ic = 0; ic < ngrid[2] ; ic++){
                        igrid[0] = ia ;
                        igrid[1] = ib ;
                        igrid[2] = ic ;

                        int index = igrid[0]*ngrid[1]*ngrid[2] + igrid[1]*ngrid[2] + igrid[2];
                        // get real space distance from atom to this grid point
                        Vec3 r_grid;
                        for(int k=0; k<3; k++)
                            r_grid[k] = PME_grid_positions[index][k];
                        // minimum image, see comments above about orthrhombic box limitation
                        Vec3 dr = ReferenceForce::getDeltaRPeriodic( r_grid , r_i , periodicBoxVectors );
                        double inverseR = 1.0 / sqrt(dr.dot(dr));
                        double alphaR = alphaEwald / inverseR ;  // just alpha * r

                        if (erf(alphaR) > 1e-6) {
                            vext_grid[index] -= ONE_4PI_EPS0*q_i*inverseR*erf(alphaR);
                        }
                        else{
                            vext_grid[index] -= alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*q_i;
                        }

                    }
                }
            }    

        }

    }



    /*   Free allocated memory from copied PME data structures */
    free(particleindex);
    // temporary data structure
    free(QMexclusion_flag);
}



/**---------------------------------------------------------------------------------------

   Calculate LJ Coulomb pair ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculatePairIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                             vector<vector<double> >& atomParameters, vector<set<int> >& exclusions,
                                             vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal,
                                             double* vext_grid, const vector<int>& QMexclude , vector<Vec3>& PME_grid_positions ) const {

    if (ewald || pme || ljpme) {
        calculateEwaldIxn(numberOfAtoms, atomCoordinates, atomParameters, exclusions, forces,
                          totalEnergy, includeDirect, includeReciprocal, vext_grid, QMexclude, PME_grid_positions);
        return;
    }
    if (!includeDirect)
        return;
    if (cutoff) {
        for (auto& pair : *neighborList)
            calculateOneIxn(pair.first, pair.second, atomCoordinates, atomParameters, forces, totalEnergy);
    }
    else {
        for (int ii = 0; ii < numberOfAtoms; ii++) {
            // loop over atom pairs

            for (int jj = ii+1; jj < numberOfAtoms; jj++)
                if (exclusions[jj].find(ii) == exclusions[jj].end())
                    calculateOneIxn(ii, jj, atomCoordinates, atomParameters, forces, totalEnergy);
        }
    }
}

/**---------------------------------------------------------------------------------------

     Calculate LJ Coulomb pair ixn between two atoms

     @param ii               the index of the first atom
     @param jj               the index of the second atom
     @param atomCoordinates  atom coordinates
     @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
     @param forces           force array (forces added)
     @param totalEnergy      total energy

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculateOneIxn(int ii, int jj, vector<Vec3>& atomCoordinates,
                                            vector<vector<double> >& atomParameters, vector<Vec3>& forces,
                                            double* totalEnergy) const {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);

    double r2        = deltaR[0][ReferenceForce::R2Index];
    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double switchValue = 1, switchDeriv = 0;
    if (useSwitch) {
        double r = deltaR[0][ReferenceForce::RIndex];
        if (r > switchingDistance) {
            double t = (r-switchingDistance)/(cutoffDistance-switchingDistance);
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
            switchDeriv = t*t*(-30+t*(60-t*30))/(cutoffDistance-switchingDistance);
        }
    }
    double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
    double sig2 = inverseR*sig;
    sig2 *= sig2;
    double sig6 = sig2*sig2*sig2;

    double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
    double dEdR = switchValue*eps*(12.0*sig6 - 6.0)*sig6;
    if (cutoff)
        dEdR += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*(inverseR-2.0f*krf*r2);
    else
        dEdR += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;
    dEdR     *= inverseR*inverseR;
    double energy = eps*(sig6-1.0)*sig6;
    if (useSwitch) {
        dEdR -= energy*switchDeriv*inverseR;
        energy *= switchValue;
    }
    if (cutoff)
        energy += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*(inverseR+krf*r2-crf);
    else
        energy += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;

    // accumulate forces

    for (int kk = 0; kk < 3; kk++) {
        double force  = dEdR*deltaR[0][kk];
        forces[ii][kk]   += force;
        forces[jj][kk]   -= force;
    }

    // accumulate energies

    if (totalEnergy)
        *totalEnergy += energy;
}

