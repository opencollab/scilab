// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) Scilab Enterprises - 2012 - Clément DAVID
//
// This file is distributed under the same license as the Scilab package.

// <-- XCOS TEST -->
// <-- ENGLISH IMPOSED -->
//
// <-- Short Description -->
// White-box test for the xcosPalGenerateAllIcons.

loadXcosLibs();

// on the standard palette
xcosPalGenerateAllIcons(["Palettes" "Recently Used Blocks"]);

// on a custom palette
pal = xcosPal();
pal = xcosPalAddBlock(pal, "BIGSOM_f", "SCI/modules/xcos/images/palettes/GOTO.png", "SCI/modules/xcos/images/palettes/GOTO.png");
assert_checktrue(xcosPalAdd(pal, "my Summation blocks"));

xcosPalGenerateAllIcons("my Summation blocks");

