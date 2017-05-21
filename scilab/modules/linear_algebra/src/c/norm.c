/*
 * Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
 * Copyright (C) 2013 - Scilab Enterprises - Paul Bignier
 *
 * Copyright (C) 2012 - 2016 - Scilab Enterprises
 *
 * This file is hereby licensed under the terms of the GNU GPL v2.0,
 * pursuant to article 5.3.4 of the CeCILL v.2.1.
 * This file was originally licensed under the terms of the CeCILL v2.1,
 * and continues to be available under such terms.
 * For more information, see the COPYING file which you should have received
 * along with this program.
 *
 */

#include <omp.h>
#include "norm.h"
#include "sci_malloc.h"

#ifdef _MSC_VER
int la_isinf(double dbl)
{
    //check finite and non NaN values
    if (_finite(dbl) == 0 && dbl == dbl)
    {
        if (dbl < 0)
        {
            return -1;
        }
        else
        {
            return 1;
        }
    }
    return 0;
}
#endif

// Lapack routines, for complex and real input
extern double C2F(dlange) (const char *norm, int *m, int *n, double *A, int *lda, double *work);
extern int     C2F(dgesdd) (const char *job, int *m, int *n, double *A, int *lda,
                            double *s, double *u, int *ldu, double *vt, int *ldvt,
                            double *work, int *lwork, int *iwork, int *info);
extern double C2F(zlange) (const char *norm, int *m, int *n, doublecomplex *A, int *lda, double *work);
extern int     C2F(zgesdd) (const char *job, int *m, int *n, doublecomplex *A, int *lda,
                            double *s, doublecomplex *u, int *ldu, doublecomplex *vt, int *ldvt,
                            doublecomplex *work, int *lwork, double *rwork, int *iwork, int *info);

double normString (double *A, int iRows, int iCols, char *flag)
{
    double ret = 0;
    double *work = NULL;

    //flag cannot be both inf and fro at same time, so use mutual exclusion using if... if else
    if (strcmp(flag, "inf") == 0 || strcmp(flag, "i") == 0)
    {
        work = (double *)CALLOC(Max(1, iRows), sizeof(double));

        // Call Lapack routine for computation of the infinite norm.
        ret = C2F(dlange)("I", &iRows, &iCols, A, &iRows, work);
        FREE(work);
    }
    else if (strcmp(flag, "fro") == 0 || strcmp(flag, "f") == 0)
    {
        // Call Lapack routine for computation of the Frobenius norm.
        ret = C2F(dlange)("F", &iRows, &iCols, A, &iRows, NULL);
    }
    return ret;
}

double normStringC (doublecomplex *A, int iRows, int iCols, char *flag)
{
    double ret = 0;
    double *work = NULL;

    //flag cannot be both inf and fro at same time, so use mutual exclusion using if... if else
    if (strcmp(flag, "inf") == 0 || strcmp(flag, "i") == 0)
    {
        work = (double *)MALLOC(Max(1, iRows) * sizeof(double));

        // Call Lapack routine for computation of the infinite norm.
        ret = C2F(zlange)("I", &iRows, &iCols, A, &iRows, work);
        FREE(work);
    }
    else if (strcmp(flag, "fro") == 0 || strcmp(flag, "f") == 0)
    {
        // Call Lapack routine for computation of the Frobenius norm.
        ret = C2F(zlange)("F", &iRows, &iCols, A, &iRows, NULL);
    }
    return ret;
}

double normP (double *A, int iRows, int iCols, double p)
{
    double ret = 0, minA, scale = 0;
    double *S, *work;
    int *iwork;
    int i, maxRC, minRC, lwork, info, one = 1;

    //Remove two function calls for assigning to iRows and iCols
    //Assign max of iRows and iCols to maxRC
    //Assign min of iRows and iCols to minRC
    if(iRows>iCols)
    {
        maxRC=iRows;
        minRC=iCols;
    }
    else
    {
        maxRC=iCols;
        minRC=iRows;
    }
    
    lwork = 3 * minRC + Max(maxRC, 7 * minRC);

    if (ISNAN(p)) // p = %nan is a special case, return 0./0 = %nan.
    {
        double a = 1.0;
        double b = 1.0;
        return ((b - a) / (a - b));
    }

    //
    // /!\ la_isinf return only 0 or 1 on non Linux platforms
    //
    if (la_isinf(p) != 0 && p < 0) // p = -%inf is a special case, return min(abs(A)).
    {
        minA = Abs(A[0]);

        //predefined min reduction supported from openMP v3.1 onwards
        #pragma omp parallel for reduction(min:minA)
        for (i = 1; i < iRows; ++i)
        {
            double absolute_Ai=Abs(A[i]);
            if(absolute_Ai<minA)
            {
                minA = absolute_Ai;
            }
        }
        return minA;
    }
    if (p == 0) // p = 0 is a special case, return 1./0 = %inf.
    {
        double a = 1.0;
        double b = 1.0;
        return (1. / (a - b));
    }
    else if (p == 1) // Call the Lapack routine for computation of norm 1.
    {
        return C2F(dlange)("1", &iRows, &iCols, A, &iRows, NULL);
        
    }
    else if (p == 2) // Call the Lapack routine for computation of norm 2.
    {
        if (iCols == 1) // In the vector case, doing a direct calculation is faster.
        {
            //predefined max reduction supported from openMP v3.1 onwards
            #pragma omp parallel for reduction(max:scale)
            for (i = 0; i < iRows; ++i)
            {
                double absolute_Ai=Abs(A[i]);
                if(absolute_Ai>scale)
                {
                    scale = absolute_Ai;
                }
            }
            
            // Eliminated equality comparison of floating point values
            if (((int)scale) == 0)
            {
                return 0;
            }
            else
            {
                #pragma omp parallel for reduction(+:ret)
                for (i = 0; i < iRows; ++i)
                {
                    double temp = A[i];
                    ret += temp * temp;
                }
                //take advantage of scale being loop invariant
                double scale_square=scale*scale;
                return scale * sqrt(ret/scale_square);
            }
        }
        // Allocating workspaces.
        S     = (double *)MALLOC(minRC * sizeof(double));
        work  = (double *)MALLOC(Max(1, lwork) * sizeof(double));
        iwork = (int *)MALLOC(8 * minRC * sizeof(int));

        // Not computing singular vectors, so arguments 7, 8, 9 and 10 are dummies.
        C2F(dgesdd)("N", &iRows, &iCols, A, &iRows, S, NULL, &one, NULL, &one, work, &lwork, iwork, &info);
        if(info==0)
        {
            // successful termination.
            // The largest singular value of A is stored in the first element of S, return it.
            return S[0];
        }
        else
        {
            //else block for clarity in semantics
            //Lapack provides it's own error messages, return
            FREE(S);
            FREE(work);
            FREE(iwork);
            return 0;
        }
    }
    // Here, A is a vector of length iRows, return sum(abs(A(i))^p))^(1/p).
    if ((int) p == p && (int) p % 2 == 0) // No need to call Abs if p is divisible by 2.
    {
        // utilize parallelism using omp and reduce sum to variable ret
        #pragma omp parallel for reduction(+:ret)
        for (i = 0; i < iRows; ++i)
        {
            ret += pow(A[i], p);
        }
    }
    else
    {
        #pragma omp parallel for reduction(+:ret)
        for (i = 0; i < iRows; ++i)
        {
            ret += pow(Abs(A[i]), p);
        }
    }
    return pow(ret, 1. / p);
}

double normPC (doublecomplex *A, int iRows, int iCols, double p)
{
    double ret = 0, minA;
    double *S, *rwork;
    doublecomplex *work;
    int *iwork;
    int i, maxRC, minRC, lwork, lrwork, info, one = 1;

    //Assign max of iRows and iCols to maxRC
    //Assign min of iRows and iCols to minRC
    if(iRows>iCols)
    {
        maxRC=iRows;
        minRC=iCols;
    }
    else
    {
        maxRC=iCols;
        minRC=iRows;
    }

    lwork  = 2 * minRC + maxRC;
    lrwork = 5 * minRC;

    if (ISNAN(p)) // p = %nan is a special case, return 0./0 = %nan.
    {
        double a = 1.0;
        double b = 1.0;
        ret = (a - b) / (a - b);        ////review
        return ret;
    }

    if (la_isinf(p) != 0 && p < 0) // p = -%inf is a special case, return min(abs(A)).
    {
        minA = sqrt(A[0].r * A[0].r + A[0].i * A[0].i); // Retrieving A[0] modulus.
        
        //predefined min reduction supported from openMP v3.1 onwards
        #pragma omp parallel for reduction(min:minA)
        for (i = 1; i < iRows; ++i)
        {
            double real=A[i].r;
            double imag=A[i].i;
            double modulusAi=sqrt(real * real + imag * imag);
            // Starting at zero in case A has only one element.
            // min(minA, modulus(A[i]))
            if(modulusAi<minA)
            {
                minA = modulusAi;
            }
        }
        return minA;
    }
    
    if (p == 0) // p = 0 is a special case, return 1./0 = %inf.
    {
        double a = 1.0;
        double b = 1.0;
        return (1. / (a - b));
    }
    else if (p == 1) // Call the Lapack routine for computation of norm 1.
    {
        return C2F(zlange)("1", &iRows, &iCols, A, &iRows, NULL);
    }
    else if (p == 2) // Call the Lapack routine for computation of norm 2.
    {
        if (iCols == 1) // In the vector case, doing a direct calculation is faster.
        {
            #pragma omp parallel for reduction(+:ret)
            for (i = 0; i < iRows; ++i)
            {
                double real=A[i].r;
                double imag=A[i].i;
                ret += real * real + imag * imag; // Retrieving A[i] modulus^2.
            }
            return sqrt(ret);
        }
        // Allocating workspaces.
        S     = (double *)MALLOC(minRC * sizeof(double));
        work  = (doublecomplex *)MALLOC(Max(1, lwork) * sizeof(doublecomplex));
        rwork = (double *)MALLOC(Max(1, lrwork) * sizeof(double));
        iwork = (int *)MALLOC(8 * minRC * sizeof(int));

        // Not computing singular vectors, so arguments 7, 8, 9 and 10 are dummies.
        C2F(zgesdd)("N", &iRows, &iCols, A, &iRows, S, NULL, &one, NULL, &one, work, &lwork, rwork, iwork, &info);
        if (info == 0)
        {
                // info = 0: successful termination.
                // The largest singular value of A is stored in the first element of S, return it.
                return S[0];
        }
        else
        {
            //else block for clarity in semantics
            //Lapack provides it's own error messages, return
            FREE(S);
            FREE(work);
            FREE(rwork);
            FREE(iwork);
            return 0;
        }
    }
    // Here, A is a vector of length iRows, return sum(abs(A(i))^p))^(1/p).
    // sum up modulus(A[i])^p into ret and return ret^(1/p)
    #pragma omp parallel for reduction(+:ret)
    for (i = 0; i < iRows; ++i)
    {
        double real=A[i].r;
        double imag=A[i].i;
        // sum(modulus(A[i])^p), same as ((A[i].r)^2+(A[i].i)^2)^(p/2)
        ret += pow(real * real + imag * imag, p / 2);
    }
    return pow(ret, 1. / p); // sum(modulus(A[i])^p)^(1/p).
}
