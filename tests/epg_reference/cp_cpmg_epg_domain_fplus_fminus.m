function [F0_vector_out,Xi_F_out,Xi_Z_out,Xi_F_all_out,Xi_Z_all_out] = cp_cpmg_epg_domain_fplus_fminus (N_in,alpha_in,ESP_in,T1_in,T2_in)
% Weigel EPG implementation for CPMG sequences
% Source: https://github.com/matthias-weigel/EPG
% License: MIT

refoc_mode  = +1;    % CPMG mode
init_90pad2 =  0;

alpha_in = alpha_in/180.0*pi;
fa       = alpha_in;

for pn = 2:N_in
    fa = [fa,alpha_in];
end

N     = length(fa);
Nt2   = 2*N;                                     
Nt2p1 = Nt2+1;  
Nt2p2 = Nt2+2;

if ((init_90pad2 == 1) && (N>1))
    fa(1) = (pi+fa(2))/2;
end
    
if (T1_in == 0)
    E1 = 1.0;
else
    E1 = exp(-ESP_in/T1_in/2.0);
end

if (T2_in == 0)
    E2 = 1.0;
else
    E2 = exp(-ESP_in/T2_in/2.0);
end

Omega_preRF  = zeros(3,Nt2p1);
Omega_postRF = zeros(3,Nt2p1);

Xi_F_all_out = zeros(4*N+1,Nt2);
Xi_Z_all_out = zeros(Nt2p1,Nt2);

if (refoc_mode == +1)
    Omega_postRF(1,1) = 1;
    Omega_postRF(2,1) = 1;
else
    Omega_postRF(1,1) = -1i;
    Omega_postRF(2,1) = +1i;
end

for pn = 1:N
    T(1,1) =         cos(fa(pn)/2)^2;   
    T(1,2) =         sin(fa(pn)/2)^2;
    T(1,3) = -1.0i * sin(fa(pn)  )  ;
    T(2,1) =         sin(fa(pn)/2)^2;
    T(2,2) =         cos(fa(pn)/2)^2;
    T(2,3) = +1.0i * sin(fa(pn)  )  ;
    T(3,1) = -0.5i * sin(fa(pn)  )  ;
    T(3,2) = +0.5i * sin(fa(pn)  )  ;           
    T(3,3) =         cos(fa(pn)  )  ;
    
    pn2 = 2*pn;
    k   = 1:(pn2-1);
    kp1 = 1:(pn2);
    kp2 = 1:(pn2+1);

    Omega_preRF(1:2,k)      = E2 * Omega_postRF(1:2,k);
    Omega_preRF(3,k(2:end)) = E1 * Omega_postRF(3,k(2:end));
    Omega_preRF(3,1)        = E1 * Omega_postRF(3,1) + 1 - E1;
        
    Omega_preRF(1,k+1) = Omega_preRF(1,k);
    Omega_preRF(2,k)   = Omega_preRF(2,k+1);
    Omega_preRF(1,1)   = conj(Omega_preRF(2,1));

    Omega_postRF(:,kp1) = T * Omega_preRF(:,kp1);

    Xi_F_all_out(Nt2p2-kp1,pn2-1)        =      Omega_postRF(1,kp1);
    Xi_F_all_out(Nt2+kp1(2:end),pn2-1) = conj(Omega_postRF(2,kp1(2:end)));
    Xi_Z_all_out(Nt2p2-kp1,pn2-1)        =      Omega_postRF(3,kp1);

    Omega_postRF(1:2,kp1)      = E2 * Omega_postRF(1:2,kp1);
    Omega_postRF(3,kp1(2:end)) = E1 * Omega_postRF(3,kp1(2:end));
    Omega_postRF(3,1)          = E1 * Omega_postRF(3,1) + 1 - E1;

    Omega_postRF(1,kp1+1) = Omega_postRF(1,kp1);
    Omega_postRF(2,kp1)   = Omega_postRF(2,kp1+1);
    Omega_postRF(1,1)     = conj(Omega_postRF(2,1));

    Xi_F_all_out(Nt2p2-kp2,2*pn)        =      Omega_postRF(1,kp2);
    Xi_F_all_out(Nt2+kp2(2:end),2*pn) = conj(Omega_postRF(2,kp2(2:end)));
    Xi_Z_all_out(Nt2p2-kp2,2*pn)        =      Omega_postRF(3,kp2);
     
end

Xi_F_all_out(abs(Xi_F_all_out)<eps*1e3) = 0;
Xi_Z_all_out(abs(Xi_Z_all_out)<eps*1e3) = 0;

indexer = 2:2:(Nt2);
Xi_F_out = Xi_F_all_out(:,indexer);
Xi_Z_out = Xi_Z_all_out(:,indexer);

F0_vector_out = Xi_F_out(Nt2p1,:);

Xi_F_out = Xi_F_out(1:2:end,:);
Xi_Z_out = Xi_Z_out(2:2:end,:);

end
