void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = f[n-1-j];
      f_m2mjpn = f[n-2-j];
      f_m2pn = f[n-2];
      f_m1pn = f[n-1];
      f_i = f[0];
      f_ipj = f[j];
      f_ip1pj = f[j+1];
      for(i=0;i<=k;++i)
      {
        f_ip1 = f[i+1];
        f_ip2pj = f[i+2+j];
        b0 += (f_ip1pj - f_ip1 - f_i)*f_ip1pj + f_ipj*(f_ip1 + f_i - f_ipj);
        b1 += ((f_ipj*f_ipj) - (f_ip2pj + f_ipj)*f_ip1) + (2*f_ip1 + (f_ip2pj - f_ipj - f_ip1pj))*f_ip1pj;
        b2 += 2*(f_i + f_ipj - f_ip1 - f_ip2pj)*f_ip1pj + f_ip2pj*(f_ip1 + f_ip2pj - f_i) + f_ipj*(f_ip1 - f_i - f_ipj);
	f_i = f_ip1;
	f_ipj = f_ip1pj;
	f_ip1pj = f_ip2pj;
      }
      b0 += -f_m2mjpn*f_m1pn + f_m2pn*(f_m2mjpn + f_m1mjpn - f_m2pn) + f_m1mjpn*(f_m1pn - f_m1mjpn);
      b1 += (f_m1mjpn*(f_m1mjpn - f_m2mjpn) + f_m2pn*(f_m2pn - f_m1mjpn - f_m1pn) + f_m1pn*f_m2mjpn);
      b2 += f_m1mjpn*(2*f_m2mjpn - f_m1mjpn + f_m2pn - f_m1pn) - f_m2pn*(f_m2pn + f_m2mjpn) - (f_m2mjpn*f_m2mjpn) + f_m1pn*(2*f_m2pn - f_m1pn + f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = 2.0*b1;
  *a2 = b2;
  *a3 = b3;
}
