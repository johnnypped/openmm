#ifndef OPENMM_REFERENCEPLATFORM_H_
#define OPENMM_REFERENCEPLATFORM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/windowsExport.h"
#include <vector>

namespace OpenMM {

/**
 * This Platform subclass uses the reference implementations of all the OpenMM kernels.
 */

class OPENMM_EXPORT ReferencePlatform : public Platform {
public:
    class PlatformData;
    ReferencePlatform();
    const std::string& getName() const {
        static const std::string name = "Reference";
        return name;
    }
    double getSpeed() const;
    bool supportsDoublePrecision() const;
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>& properties) const;
    void contextDestroyed(ContextImpl& context) const;

     /* This is the name of the parameter for selecting whether to grid external potential.      */
    static const std::string& ReferenceVextGrid() {
        static const std::string key = "ReferenceVextGrid";
        return key;
    }



};

class OPENMM_EXPORT ReferencePlatform::PlatformData {
public:
    PlatformData(const System& system, bool ReferenceVextGrid);
    ~PlatformData();
    int numParticles, stepCount;
    double time;
    void* positions;
    void* velocities;
    void* forces;
    void* periodicBoxSize;
    void* periodicBoxVectors;
    void* constraints;
    void* energyParameterDerivatives;

    std::map<std::string, std::string> propertyValues;

    // QMatoms to exclude in vext_grid calculation
    const std::vector<int>& QMexclude;  // does not own, owned by system

    // this stores external potential evaluated on PME grid //
    double* vext_grid;
    // positions of PME grid points (for interpolation)
    void* PME_grid_positions;

};
} // namespace OpenMM

#endif /*OPENMM_REFERENCEPLATFORM_H_*/
