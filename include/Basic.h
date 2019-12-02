#ifndef Basic_H
#define Basic_H

#include <math.h>

#include "PrecisionTypes.h"

/** method to calculate the parallel dot product with vect1, vect2 having the ghost cells*/
double dotP(FPfield *vect1, FPfield *vect2, int n);
/** method to calculate dot product */
double dot(FPfield *vect1, FPfield *vect2, int n);
/** method to calculate the square norm of a vector */
double norm2(FPfield **vect, int nx, int ny);
/** method to calculate the square norm of a vector */
double norm2(FPfield ***vect, int nx, int ny);
/** method to calculate the square norm of a vector */
double norm2(FPfield *vect, int nx);



/** method to calculate the parallel dot product */
double norm2P(FPfield ***vect, int nx, int ny, int nz);
/** method to calculate the parallel norm of a vector on different processors with the ghost cell */
double norm2P(FPfield *vect, int n);
/** method to calculate the parallel norm of a vector on different processors with the gost cell*/
double normP(FPfield *vect, int n);
/** method to calculate the difference of two vectors*/
void sub(FPfield *res, FPfield *vect1, FPfield *vect2, int n);
/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(FPfield *vect1, FPfield *vect2, int n);
/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(FPfield  ***vect1, FPfield ***vect2, int nx, int ny, int nz);

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(FPfield  ***vect1, FPfield  ***vect2, int nx, int ny);

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(FPfield  ***vect1, FPfield ****vect2, int nx, int ny, int nz, int ns);

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(FPfield  ***vect1, FPfield  ****vect2, int nx, int ny, int ns);
/** method to calculate the subtraction of two vectors vector1 = vector1 - vector2*/
void sub(FPfield ***vect1, FPfield ***vect2, int nx, int ny, int nz);

/** method to calculate the subtraction of two vectors vector1 = vector1 - vector2*/
void sub(FPfield ***vect1, FPfield ***vect2, int nx, int ny);


/** method to sum 4 vectors vector1 = alfa*vector1 + beta*vector2 + gamma*vector3 + delta*vector4 */
void sum4(double ***vect1, double alfa, double ***vect2, double beta, double ***vect3, double gamma, double ***vect4, double delta, double ***vect5, int nx, int ny, int nz);
/** method to calculate the scalar-vector product */
void scale(FPfield *vect, FPfield  alfa, int n);

/** method to calculate the scalar-vector product */
void scale(FPfield  ***vect, FPfield  alfa, int nx, int ny);

/** method to calculate the scalar-vector product */
void scale(FPfield  ***vect, FPfield alfa, int nx, int ny, int nz);
/** method to calculate the scalar product */
void scale(double vect[][2][2], double alfa, int nx, int ny, int nz);
/** method to calculate the scalar-vector product */
void scale(FPfield  ***vect1, FPfield ***vect2, FPfield  alfa, int nx, int ny, int nz);

/** method to calculate the scalar-vector product */
void scale(FPfield  ***vect1, FPfield  ***vect2, double alfa, int nx, int ny);

/** method to calculate the scalar-vector product */
void scale(FPfield *vect1, FPfield *vect2, double alfa, int n);
/** method to calculate vector1 = vector1 + alfa*vector2   */
void addscale(FPfield alfa, FPfield ***vect1, FPfield ***vect2, int nx, int ny, int nz);
/** add scale for weights */
void addscale(double alfa, double vect1[][2][2], double vect2[][2][2], int nx, int ny, int nz);
/** method to calculate vector1 = vector1 + alfa*vector2   */
void addscale(FPfield alfa, FPfield ***vect1, FPfield ***vect2, int nx, int ny);
/** method to calculate vector1 = vector1 + alfa*vector2   */
void addscale(FPfield alfa, FPfield *vect1, FPfield *vect2, int n);
/** method to calculate vector1 = beta*vector1 + alfa*vector2   */
void addscale(FPfield alfa, FPfield beta, FPfield *vect1, FPfield *vect2, int n);
/** method to calculate vector1 = beta*vector1 + alfa*vector2 */
void addscale(FPfield alfa, FPfield beta, FPfield ***vect1, FPfield ***vect2, int nx, int ny, int nz);
/** method to calculate vector1 = beta*vector1 + alfa*vector2 */
void addscale(double alfa, double beta, double ***vect1, double ***vect2, int nx, int ny);


/** method to calculate vector1 = alfa*vector2 + beta*vector3 */
void scaleandsum(double ***vect1, double alfa, double beta, double ***vect2, double ***vect3, int nx, int ny, int nz);
/** method to calculate vector1 = alfa*vector2 + beta*vector3 with vector2 depending on species*/
void scaleandsum(double ***vect1, double alfa, double beta, double ****vect2, double ***vect3, int ns, int nx, int ny, int nz);
/** method to calculate vector1 = alfa*vector2*vector3 with vector2 depending on species*/
void prod(double ***vect1, double alfa, double ****vect2, int ns, double ***vect3, int nx, int ny, int nz);
/** method to calculate vect1 = vect2/alfa */
void div(double ***vect1, double alfa, double ***vect2, int nx, int ny, int nz);
void prod6(double***, double***, double***, double***, double***, double***, double***, int, int, int);
/** method used for calculating PI */
void proddiv(double***, double***, double, double***, double***, double***, double***, double, double***, double***, double, double***, int, int, int);
/** method to calculate the opposite of a vector */
void neg(FPfield***, int, int, int);

/** method to calculate the opposite of a vector */
void neg(FPfield***, int, int);
/** method to calculate the opposite of a vector */
void neg(FPfield***, int);
/** method to calculate the opposite of a vector */
void neg(FPfield*, int);
/** method to set equal two vectors */
void eq(double***, double***, int, int, int);
/** method to set equal two vectors */
void eq(double***, double***, int, int);

/** method to set equal two vectors */
void eq(double****, double***, int, int, int);
/** method to set equal two vectors */
void eq(double****, double***, int, int, int, int);

void eq(double*, double*, int);
/** method to set a vector to a Value */
void eqValue(FPfield, FPfield***, int, int, int);
void eqValue(double, double[][2][2], int, int, int);
/** method to set a vector to a Value */
void eqValue(double, double***, int, int);
/** method to set a vector to a Value */
void eqValue(double, double***, int);
/** method to set a vector to a Value */
void eqValue(FPfield, FPfield*, int);
/** method to put a column in a matrix 2D */
void putColumn(FPfield**, FPfield*, int, int);
/** method to get a column in a matrix 2D */
void getColumn(FPfield*, FPfield**, int, int);
/** RIFAI QUESTA PARTE questo e' la tomba delle performance*/
void MODULO(double*, double);
/** method to calculate the epsilon machine */
double eps();


#endif
