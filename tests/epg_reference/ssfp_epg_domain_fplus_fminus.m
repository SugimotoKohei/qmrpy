function [F0_vector_out,Xi_F_out,Xi_Z_out] = ssfp_epg_domain_fplus_fminus (N_in,alpha_in,TR_in,T1_in,T2_in)
% Weigel EPG implementation for SSFP sequences
% Source: https://github.com/matthias-weigel/EPG
% License: MIT

% Hard coded settings - FLASH / RF spoiled GE
unbalanced    =  1;
RF_phase_mode =  1;
RF_phase_inc  = 50;

alpha_in = alpha_in/180.0*pi;
fa       = alpha_in;

for pn = 2:N_in
    fa = [fa,alpha_in];
end

N   = length(fa);
Nm1 = N-1;                                     
Np1 = N+1;  

switch (RF_phase_mode)
    case -1
        fa(2:2:end) = -fa(2:2:end);
        phi (1:N)    = 0;
        if (N>1)
            fa(1)=fa(1)/2;
        end
        
    case +1
        pn  = 1:N;
        phi = (pn-1).*pn/2.0 * RF_phase_inc / 180.0 * pi;        
        
    otherwise
        phi(1:N) = 0;
end
    
E1 = exp(-TR_in/T1_in);
E2 = exp(-TR_in/T2_in);

Omega_preRF  = zeros(3,Np1);
Omega_postRF = zeros(3,N);

Xi_F_out = zeros(2*N-1,N);
Xi_Z_out = zeros(N,N);

Omega_preRF(3,1) = 1;    

for pn = 1:N
    T(1,1) =                            cos(fa(pn)/2)^2;   
    T(1,2) =         exp(+2i*phi(pn)) * sin(fa(pn)/2)^2;
    T(1,3) = -1.0i * exp(+1i*phi(pn)) * sin(fa(pn)  )  ;
    T(2,1) =         exp(-2i*phi(pn)) * sin(fa(pn)/2)^2;
    T(2,2) =                            cos(fa(pn)/2)^2;
    T(2,3) = +1.0i * exp(-1i*phi(pn)) * sin(fa(pn)  )  ;
    T(3,1) = -0.5i * exp(-1i*phi(pn)) * sin(fa(pn)  )  ;
    T(3,2) = +0.5i * exp(+1i*phi(pn)) * sin(fa(pn)  )  ;           
    T(3,3) =                            cos(fa(pn)  )  ;
    
    k = 1:pn;
    
    Omega_postRF(:,k) = T * Omega_preRF(:,k);

    Xi_F_out(Np1-k,pn)        =      Omega_postRF(1,k);
    Xi_F_out(Nm1+k(2:end),pn) = conj(Omega_postRF(2,k(2:end)));
    Xi_Z_out(Np1-k,pn)        =      Omega_postRF(3,k);

    Omega_preRF(1:2,k)      = E2 * Omega_postRF(1:2,k);
    Omega_preRF(3,k(2:end)) = E1 * Omega_postRF(3,k(2:end));
    Omega_preRF(3,1)        = E1 * Omega_postRF(3,1) + 1 - E1;
    
    if (unbalanced == 1)      
        Omega_preRF(1,k+1) = Omega_preRF(1,k);
        Omega_preRF(2,k)   = Omega_preRF(2,k+1);
        Omega_preRF(1,1)   = conj(Omega_preRF(2,1));
    end
  
end

Xi_F_out(abs(Xi_F_out)<eps*1e3) = 0;
Xi_Z_out(abs(Xi_Z_out)<eps*1e3) = 0;

F0_vector_out = Xi_F_out(N,:);

end
