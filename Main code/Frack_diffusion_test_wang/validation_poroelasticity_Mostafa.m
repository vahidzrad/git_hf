clc
clear
%close all

perm=1e-18; muf=1e-3; niu=0.34; E=6e9; r0=10e-3; p0=1e6; s1=0.; s2=0.; L=200e-3; theta=pi/2;
G=E/2/(1+niu);
kappa=perm/muf;
ita=(1-2*niu)/2/(1-niu);
c=2*kappa*G*(1-niu)/(1-2*niu);
%R=r0:r0/2:r0*3/2;
R=[];
P=[]; SR=[]; ST=[];
t=0.1;
for r=r0:r0/18/2:r0*1.25;
    R=[R (r-r0)/r0];
    
    sr1=(s1+s2)/2*(1-r0^2/r^2)-(s1-s2)/2*(1-4*r0^2/r^2+3*r0^4/r^4)*cos(2*theta);
    sr2=-p0*r0^2/r^2;
    sr3=-p0*(-2*ita*(r0/r)^1.5*(sqrt(4*c*t/pi/r0^2)*exp(-(r-r0)^2/4/c/t)-(r/r0-1)*erfc(((r-r0)/sqrt(4*c*t))))+2*ita*(r0/r)^2*sqrt(4*c*t/pi/r0^2));
    SR=[SR sr1+sr2+sr3];
    
    st1=(s1+s2)/2*(1+r0^2/r^2)+(s1-s2)/2*(1+3*r0^4/r^4)*cos(2*theta);
    st2=p0*r0^2/r^2;
    st3=-p0*2*ita*((r0/r)^1.5*(sqrt(4*c*t/r0^2/pi)*exp(-(r-r0)^2/4/c/t)+erfc((r-r0)/sqrt(4*c*t)))-r0^2/r^2*sqrt(4*c*t/r0^2/pi)...
        +1/8*sqrt(r0/r)*(1-r0/r)*(sqrt(4*c*t/r0^2/pi)*exp(-(r-r0)^2/4/c/t)-(r/r0-1)*erfc((r-r0)/sqrt(4*c*t))));
    ST=[ST -st3];
    
    p1=0;
    p2=p0;
    p3=p0*(-1+sqrt(r0./r).*erfc((r-r0)/sqrt(4*c*t))+1/8*sqrt(r0/r)*(1-r0/r)*(sqrt(4*c*t/pi/r0^2)*exp(-(r-r0)^2/4/c/t)-(r/r0-1)*erfc((r-r0)/sqrt(4*c*t))));
    P=[P p1+p2+p3];
end

 figure(4);plot(R,-ST+P,'or','Linewidth',4)
 hold on

 figure(4);plot(R,P,'*g','Linewidth',1.5)
 hold on
%figure;plot(R,SR)