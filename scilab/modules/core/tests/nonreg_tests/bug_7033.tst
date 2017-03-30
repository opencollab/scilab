// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2010 - DIGITEO - Bernard HUGUENEY
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- Non-regression test for bug 7033 -->
//
// <-- Bugzilla URL -->
// http://bugzilla.scilab.org/show_bug.cgi?id=7033
//
// <-- Short Description -->
// Random crash (more often in 64 bits) in sci_newfun / getMatrixOfString
// trying to write to unallocated memory.

// <-- CLI SHELL MODE -->
// <-- NO CHECK REF -->

exec SCI/modules/core/tests/unit_tests/newfun.tst;
