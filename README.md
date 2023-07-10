# PP_high_dim

============== Part 1: Readme description for Matlab codes =============================

Readme Description for Producing Figure 2 (left Panel and Right Panel) and Figure 1 (Left Panel) of the paper titled "A Computational Perspective on Projection Pursuit in High Dimensions: Feasible or Infeasible Feature Extraction" by Zhang, Ye and Wang (2023, International Statistical Review, Volume 91, Issue 1, 140–161, available at https://onlinelibrary.wiley.com/doi/10.1111/insr.12517). 

Inputs such as sample size $n$, data dimension $p$, and types of data distribution, used in the codes for other figures can be adjusted manually in a similar way.

All Matlab codes are located in the same directory.

There are two methods to implement the code:

Method 1: 
(1) Main matlab code: "BKN_Theorems_1_2i.m" (attached in Part 2). 

(2) This main code calls other Mtlab functions (attached in Part 2) listed below:

est_pdf_from_histogram.m                    
CDF_mixture_Gaussians_D_dim_func.m          mean_cov_mixture_Gaussians_D_dim.m          
EDF_D_dim_func.m                            mixture_Gaussians_D_dim_generator.m         
KS_stat_1_sample.m                          pdf_mixture_Gaussians_D_dim_func.m          
Least_squares_min_w_quad_equa_constraint.m  z_vector_exist_dep_on_X_G.m                 
X_matrix_sim.m                              z_vector_unit_Euclidean_norm.m              
discrete_label_generator.m            
   
Method 2: Directly run the source code "demo_code.m" available on GitHub at https://github.com/ChunmingZhangUW/PP_high_dim.

============== Part 2: Scripts for Matlab codes =============================

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 1 &&&&&&&&&&&&&&&&&&&&&&

%
% Name    : BKN_Theorems_1_2i.m
% Function: Simulation of projection persuit in BKN (2018) with p >> n,
%           for Theorem 1, and Theorem 2(i).
%--------------------------------------------------------------------------

clear;
close all;

rng(12314357, 'twister');

%=============== Illustrate Theorem 1, Theorem 2(i) =======================
choice_G = 1;  % mixture Gaussian
choice_CDF_in_target = 'G'; % target CDF = G

if     choice_G == 1 % mixture Gaussian
    num_obs = 100;  dim_p = 1000;
end
gamma = dim_p / num_obs;

num_sim = 100; % # of Monte-Carlo runs, for boxplots of |\hat G_z - G^*|_\infty

num_grids = 100; % # of grid points;

num_bins = 10; % # of bins in the empirical pdf estimator

choice_F = 0;
if     choice_F == 0
    A = 0; B = 1; mu_F = A; mu_F_centered = 0;
    % elseif choice_F == 1
    %     A = -3; B = 3; mu_F = (A+B)/2; mu_F_centered = 0;
    % elseif choice_F == 2
    %     A = 1; B = 2; mu_F = A/(A+B); mu_F_centered = 0;
end

choice_a_vector = 1;
if choice_a_vector == 1; num_samplings = 10000; end

%==========================================================================
if     choice_G == 1 % mixture Gaussian
    prop_vector = [1/2; 1/2];

    mu = 2; sigma = 1/2;
    mu_matrix = [-mu  +mu]; sigma_matrix = [sigma  sigma];

    if strcmp(choice_CDF_in_target, 'G') == 1
        pdf_1D_in_target = @(x) pdf_mixture_Gaussians_D_dim_func...
            (prop_vector, mu_matrix, sigma_matrix, x); % pdf of target CDF = G

        CDF_1D_in_target = @(x) CDF_mixture_Gaussians_D_dim_func...
            (prop_vector, mu_matrix, sigma_matrix, x); % CDF of target CDF = G
    end

    [mu_G_vector, cov_G_matrix] = mean_cov_mixture_Gaussians_D_dim...
        (prop_vector, mu_matrix, sigma_matrix);
    disp(' mu_G_vector, cov_G_matrix are: ')
    disp( [mu_G_vector, cov_G_matrix] )

    mu_2_G = cov_G_matrix + mu_G_vector^2;

    %----------------------------------------------------
    disp('===== For Theorem 1 =====');
    disp([' gamma = p/n = ', num2str(gamma), ...
        ', mu_2(G) = ', num2str(mu_2_G)])

    disp('===== For Theorem 2(i) =====');
    if gamma > 1 && mu_2_G < gamma - 1
        disp('gamma > 1 and mu_2(G) < gamma - 1; satisfies condition in Theorem 2(i)')
    else
        disp('gamma <= 1 or mu_2(G) >= gamma - 1; violates condition in Theorem 2(i). Stop!')
        return
    end
end

D_n_exist   = zeros(num_sim, 1);
D_n_dep_2   = zeros(num_sim, 1);
D_n_indep_3 = zeros(num_sim, 1);
D_n_crt     = zeros(num_sim, 1);
tic
for sim = 1:num_sim
    disp([' sim = ', num2str(sim)])
    %--------------------- generate data matrix ---------------------------
    [X_matrix] = X_matrix_sim(choice_F, dim_p, num_obs, mu_F, A, B);
    % dim_p * num_obs matrix

    %--------------- get a_vector_G from G --------------------------------

    if choice_G == 1
        if     choice_a_vector == 1
            samples_G = mixture_Gaussians_D_dim_generator...
                (prop_vector, mu_matrix, sigma_matrix, num_samplings);
            % 1* num_samplings, vector, random samples from G

            a_vector_G = prctile(samples_G, 100*(1:num_obs)'/(num_obs+1));
            % n*1, vector, sample quantiles of G
        end
    end

    %--------------- a data-dependent direction ---------------------------
    if sim == 1;  opt_display = 1;  else;  opt_display = 0;  end
    [z_vector_data_exist] = z_vector_exist_dep_on_X_G...
        (X_matrix, a_vector_G, opt_display);

    %--------------- another data-dependent direction ---------------------

    z_vector_data_dep_2 = z_vector_unit_Euclidean_norm...
        ( mean(X_matrix(:, (1:num_obs)), 2) );
    % p*1 vector, on the unit sphere

    %-------------- a data-independent direction --------------------------

    z_vector_data_indep_3 = z_vector_unit_Euclidean_norm...
        ( normrnd(0, 1, dim_p, 1) );
    % p*1 vector, uniformly distributed on the unit sphere

    %--------------- a data-dependent critical direction ------------------
    %z_vector_crt = zeros(dim_p, 1);

    method_LS_m_w_q_q_c = 1; % for SVD method in Golub & Van Loan (2013)
    if     method_LS_m_w_q_q_c == 1
        z_vector_initial = [];
        num_randsearch_crt = [];
    end
    [z_vector_crt, ~] = ...
        Least_squares_min_w_quad_equa_constraint...
        (X_matrix', a_vector_G, 1, method_LS_m_w_q_q_c);
    % p*1 vector, on the unit sphere

    %------------------ vector S ------------------------------------------

    S_vector_exist   = (z_vector_data_exist'   * X_matrix)';
    S_vector_dep_2   = (z_vector_data_dep_2'   * X_matrix)';
    S_vector_indep_3 = (z_vector_data_indep_3' * X_matrix)';
    S_vector_crt     = (z_vector_crt'          * X_matrix)';
    % (S_1,...,S_n)', n*1 vector

    %========== Part 1: plots of pdf and CDF via 1 simulated data =========
    if sim == 1
        %--------------- 1D plot -----------------------------------------
        x_grid = linspace(-4, 4, num_grids)';

        pdf_1D_in_target_grid = zeros(num_grids, 1);
        CDF_1D_in_target_grid = zeros(num_grids, 1);
        EDF_grid_data_exist   = zeros(num_grids, 1);
        EDF_grid_data_dep_2   = zeros(num_grids, 1);
        EDF_grid_data_indep_3 = zeros(num_grids, 1);
        EDF_grid_crt          = zeros(num_grids, 1);
        for g = 1:num_grids
            pdf_1D_in_target_grid(g) = pdf_1D_in_target(x_grid(g));
            CDF_1D_in_target_grid(g) = CDF_1D_in_target(x_grid(g));
            EDF_grid_data_exist(g)   = EDF_D_dim_func(...
                S_vector_exist,   x_grid(g));
            EDF_grid_data_dep_2(g)   = EDF_D_dim_func(...
                S_vector_dep_2,   x_grid(g));
            EDF_grid_data_indep_3(g) = EDF_D_dim_func(...
                S_vector_indep_3, x_grid(g));
            EDF_grid_crt(g)          = EDF_D_dim_func(...
                S_vector_crt,     x_grid(g));
        end

        [x_data_exist,   Epdf_data_exist] = ...
            est_pdf_from_histogram(S_vector_exist,   num_bins);
        [x_data_dep_2,   Epdf_data_dep_2] = ...
            est_pdf_from_histogram(S_vector_dep_2,   num_bins);
        [x_data_indep_3, Epdf_data_indep_3] = ...
            est_pdf_from_histogram(S_vector_indep_3, num_bins);
        [x_data_crt,     Epdf_crt] = ...
            est_pdf_from_histogram(S_vector_crt,     num_bins);

        %----------------------- compare pdfs ---------
        h_1 = figure(1);
        subplot(2,2,1)
        plot(x_grid,         pdf_1D_in_target_grid, 'k-');
        hold on;
        plot(x_data_exist,   Epdf_data_exist,   'b:', 'LineWidth', 1.5);
        plot(x_data_dep_2,   Epdf_data_dep_2,   'r--', 'LineWidth', 1.0);
        plot(x_data_indep_3, Epdf_data_indep_3, 'magenta-.');
        plot(x_data_crt,     Epdf_crt,          'cyan-o', 'LineWidth', 0.8);

        x_min_max = xlim; % get the x-axis limits for the current axes.

        xlabel('{\boldmath{$s$}}', 'interpreter', 'latex');
        ylabel('{\textbf{compare pdf}}', 'interpreter', 'latex');
        title(['{\boldmath$n = ', num2str(num_obs), ...
            ',  p = ', num2str(dim_p), '$}'], 'interpreter', 'latex')

        figure_name_1 = ['BKN_Theorems_1_2i',...
            '_G=',num2str(choice_G),'_F=',num2str(choice_F),...
            '_a=',num2str(choice_a_vector),...
            '_Epdf_B=', num2str(num_bins), ...
            '_n=',num2str(num_obs),'_p=',num2str(dim_p)];
        print(h_1, '-depsc', [figure_name_1, '.eps']);

        %----------------------- compare KDEs ---------
        [KDE_data_exist,  x_data_exist_KDE] = ksdensity(S_vector_exist);

        h_5 = figure(5);
        subplot(2,2,1)
        plot(x_grid,           pdf_1D_in_target_grid, 'k-');
        hold on;
        plot(x_data_exist,     Epdf_data_exist,  'b:', 'LineWidth', 1.5);
        plot(x_data_exist_KDE, KDE_data_exist,   'r--'); hold on;

        xlabel('{\boldmath{$s$}}', 'interpreter', 'latex');
        ylabel('{\textbf{compare pdf}}', 'interpreter', 'latex');
        title(['{\boldmath$n = ', num2str(num_obs), ...
            ',  p = ', num2str(dim_p), '$}'], 'interpreter', 'latex')

        xlim([...
            min([x_min_max(1), x_data_exist_KDE]) ...
            max([x_min_max(2), x_data_exist_KDE])]);
        % set the x-axis limits for the current axes.

        figure_name_5 = ['BKN_Theorems_1_2i',...
            '_G=',num2str(choice_G),'_F=',num2str(choice_F),...
            '_a=',num2str(choice_a_vector),...
            '_KDE_Epdf_B=', num2str(num_bins), ...
            '_n=',num2str(num_obs),'_p=',num2str(dim_p)];
        print(h_5, '-depsc', [figure_name_5, '.eps']);

        %----------------------------------------------------
        disp([ ...
            mean(S_vector_exist)   mean(S_vector_dep_2)  ...
            mean(S_vector_indep_3) mean(S_vector_crt)])
    end % if sim == 1

    %-------------------- KS statistics ---------------------------------

    vector_G_S_exist   = zeros(num_obs, 1); % (G(S_1),...,G(S_n))', n*1 vector
    vector_G_S_dep_2   = zeros(num_obs, 1); % (G(S_1),...,G(S_n))', n*1 vector
    vector_G_S_indep_3 = zeros(num_obs, 1); % (G(S_1),...,G(S_n))', n*1 vector
    vector_G_S_crt     = zeros(num_obs, 1); % (G(S_1),...,G(S_n))', n*1 vector
    if     choice_G == 1 % 1-dim Gaussian mixture  of 2-components
        for i = 1:num_obs
            vector_G_S_exist(i)   = CDF_1D_in_target(S_vector_exist(i));
            vector_G_S_dep_2(i)   = CDF_1D_in_target(S_vector_dep_2(i));
            vector_G_S_indep_3(i) = CDF_1D_in_target(S_vector_indep_3(i));
            vector_G_S_crt(i)     = CDF_1D_in_target(S_vector_crt(i));
        end
    end

    D_n_exist(sim)   = KS_stat_1_sample(S_vector_exist,   vector_G_S_exist);
    D_n_dep_2(sim)   = KS_stat_1_sample(S_vector_dep_2,   vector_G_S_dep_2);
    D_n_indep_3(sim) = KS_stat_1_sample(S_vector_indep_3, vector_G_S_indep_3);
    D_n_crt(sim)     = KS_stat_1_sample(S_vector_crt,     vector_G_S_crt);
end
disp('')
disp('======== Part 2: compare box plots of KS-statistics ======')
disp([mean(D_n_exist) mean(D_n_dep_2) mean(D_n_indep_3) ...
    mean(D_n_crt)])

toc

h_3 = figure(3); %----- compare box plots of KS-statistics ------
subplot(2,2,1)
boxplot([D_n_exist, D_n_dep_2, D_n_indep_3, D_n_crt])
set(gca, 'XTickLabel', {'z_1 (exist)', 'z_2', 'z_3', 'z_crt'}, ...
    'XTickLabelRotation',0);

ylabel(['\textbf{compare} ', ...
    '{\boldmath{$\|\widehat{G}_{z}-G\|_{\infty}$}}'], 'interpreter', 'latex')
title(['{\boldmath$n = ', num2str(num_obs), ...
    ',  p = ', num2str(dim_p), '$}'], 'interpreter', 'latex');

max_D_n = max([max(D_n_exist), max(D_n_dep_2), max(D_n_indep_3), ...
    max(D_n_crt)]);
ylim([ 0-0.025 min((max(0.55, max_D_n) + 0.05), 1) ])

figure_name_3 = ['BKN_Theorems_1_2i',...
    '_G=',num2str(choice_G),'_F=',num2str(choice_F),...
    '_a=',num2str(choice_a_vector),...
    '_KS',...
    '_sims=',num2str(num_sim), ...
    '_n=',num2str(num_obs),'_p=',num2str(dim_p)];
print(h_3, '-depsc', [figure_name_3, '.eps']);


%&&&&&&&&&&&&&&&&&&&&&& scripts for code 2 &&&&&&&&&&&&&&&&&&&&&&

function CDF_val = CDF_mixture_Gaussians_D_dim_func...
    (prop_vector, mu_matrix, sigma_matrix, x_vector)

%-------------------------------------------------------------------------
% Function: compute the CDF of a D-dim Gaussian mixture of K-components
%-------------------------------------------------------------------------
% <Input>:
% prop_vector : = (p_1,...,p_K)': K*1 vector
% mu_matrix   : = (mu_1_vector,...,mu_K_vector), D*K matrix,
%   where mu_1_vector = (mu_11,...,mu_D1)',...
% sigma_matrix: = (sigma_1_vector,...,sigma_K_vector): D*K matrix
%  x_vector   : D*1 vector, (x_1,...,x_D)'
%-------------------------------------------------------------------------
% <Output>:
% CDF_val: scalar
%-------------------------------------------------------------------------

[D, K] = size(mu_matrix);

F_vector = zeros(K, 1); % F(1),...F(K)
Phi_vector = zeros(D, 1); % Phi(1),...Phi(D)
for k = 1:K
    for d = 1:D
        x_d = x_vector(d);
        mu_d_k = mu_matrix(d, k);
        sigma_d_k = sigma_matrix(d, k);

        Phi_vector(d) = normcdf( (x_d-mu_d_k)/sigma_d_k, 0, 1 );
    end

    F_vector(k) = prod(Phi_vector);
end

CDF_val = sum(prop_vector .* F_vector); % scalar

end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 3 &&&&&&&&&&&&&&&&&&&&&&

function L_vector = discrete_label_generator(prop_vector, num_obs)

% Function: generate n iid label random variables, with the p.m.f.
%  1   2  ...  K
% p_1 p_2 ... p_K, with p_1 + p_2 + ... + p_K = 1
%-------------------------------------------------------------------------
% <Input>:
% prop_vector: = (p_1,...,p_K)': K*1 vector
%   num_obs  :  n
%-------------------------------------------------------------------------
% <Output>:
% L_vector: = (X_1,...,X_n)', n*1 vector
%-------------------------------------------------------------------------

K = length(prop_vector);
prop_add_vector = [0; prop_vector]; % (K+1)*1 vector

U_vector = unifrnd(0, 1, num_obs, 1); % n*1 vector

L_vector = zeros(num_obs, 1);
for i = 1:num_obs
    for k = 1:K
        F_k_minus_1 = sum(prop_add_vector(1: (k)));
        F_k = F_k_minus_1 + prop_add_vector(k+1);

        if F_k_minus_1 < U_vector(i) && U_vector(i) <= F_k
            L_vector(i) = k;
        end
    end
end
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 4 &&&&&&&&&&&&&&&&&&&&&&

function EDF_func = EDF_D_dim_func(X_matrix, x_vector)

%-------------------------------------------------------------------
% Function: compute the EDF of n random vectors in X_matrix
%-------------------------------------------------------------------
% <Inputs>:
% X_matrix: D*n matrix, (X(:,1),...,X(:,n))
% x_vector: D*1 vector, (x_1,...,x_D)'
%-------------------------------------------------------------------
% <Outputs>:
% EDF_func: scalar
%------------------------------------------------------

[~, num_obs] = size(X_matrix);

vector_ind = zeros(num_obs, 1); % n*1, vector
for i = 1:num_obs
    vector_ind(i) = prod(X_matrix(:,i) <= x_vector); % 0 or 1
end

EDF_func = mean( vector_ind ); % scalar
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 5 &&&&&&&&&&&&&&&&&&&&&&

function [bin_centers_vector, hat_f_vector] = ...
    est_pdf_from_histogram(X_vector, num_bins)

%-------------------------------------------------------------------
% Function: density estimates from Histogram counts of X_vector
%-------------------------------------------------------------------
% <Inputs>:
% X_vector: n*1 vector
% num_bins: B, scalar
%-------------------------------------------------------------------
% <Outputs>:
% bin_centers_vector: 1*B vector
% hat_f_vector: 1*B vector
%-------------------------------------------------------------------

num_obs = length(X_vector); % n

[bin_counts_vector, bin_edges_vector] = histcounts(X_vector, num_bins);
bin_centers_vector = ...
    ( bin_edges_vector(2:end) + bin_edges_vector(1:(end-1)))/2;
% (n_1,...,n_B); (c_1,...,c_B)

rel_freq_vector = bin_counts_vector'/num_obs;
% (f_1,...,f_B) = (n_1,...,n_B)/n

hat_f_vector = rel_freq_vector/(bin_centers_vector(2)-bin_centers_vector(1));
% (\hat f(c_1),..., \hat f(c_B))
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 6 &&&&&&&&&&&&&&&&&&&&&&

function [D_n] = KS_stat_1_sample(vector_X, vector_G_X)

%--------------------------------------------------------------------------
% Funcution: compute Kolmogorov-Smirnov one-sample goodness-of-fit statistics,
% D_n = sup_{x \in R^1} |EDF(x)-G(x)|, where
%    EDF(x) is the E.D.F. of (X_1,...,X_n), and $G$ is a specified CDF.
%--------------------------------------------------------------------------
% <Inputs>:
% vector_X  : n*1 vector, (X_1,...,X_n)'
% vector_G_X: n*1 vector, (G(X_1),..., G(X_n))'
%--------------------------------------------------------------------------
% <Outputs>:
% D_n: scalar
%--------------------------------------------------------------------------

n_obs = length(vector_X);

vector_ordered_G_X = sort(vector_G_X, 'ascend');
% sorted data, G(X_{(1)}) <= ... <= G(X_{(n)})

D_n_plus  = max( (1:n_obs)'/n_obs - vector_ordered_G_X );
D_n_minus = max( vector_ordered_G_X - ((1:n_obs)-1)'/n_obs );
D_n = max( [D_n_plus, D_n_minus, 0] );

end


%&&&&&&&&&&&&&&&&&&&&&& scripts for code 7 &&&&&&&&&&&&&&&&&&&&&&

function [x_vector_opt, hat_lambda] = ...
    Least_squares_min_w_quad_equa_constraint...
    (A_matrix, b_vector, a_circle, method)

%--------------------------------------------------------------------------
% Function: compute
% x_{opt}
%=\arg\min_{x_vector: |x_vector|_2 = a_circle} \|x_vector' A_matrix'-b_vector'\|_2
%=\arg\min_{x_vector: |x_vector|_2 = a_circle} \|A_matrix x_vector-b_vector
%\|_2.
% Called: hat_x_vector_lambda.m, function_of_lambda.m
%--------------------------------------------------------------------------
% <Inputs>:
% A_matrix: matrix, n*p
% b_vector: vector, n*1
% a_circle: >0, scalar constraint in |x_vector|_2 = a_circle, e.g., a_circle = 1
% method:   1 for SVD method in Golub & Van Loan (2013);
%--------------------------------------------------------------------------
% <Outputs>:
% x_vector_opt: vector, p*1, on the unit sphere
% hat_lambda  : Lagrange multiplier for method = 1; [] for method = 2
%--------------------------------------------------------------------------

hat_lambda = [];

if method == 1 % for SVD method in Golub & Van Loan (2013)
    %svd    Singular value decomposition.
    %[U,S,V] = svd(X) produces a diagonal matrix S, of the same
    %dimension as X and with nonnegative diagonal elements in
    %decreasing order, and unitary matrices U and V so that X = U*S*V'.
    [U, Sigma, V] = svd(A_matrix); % [U,S,V] = svd(A)
    % U: n*n; Sigma: n*p; V: p*p.
    rank_A = rank(A_matrix); % r

    if sum(abs(b_vector)) == 0 % b_vector = 0

        x_vector_opt = abs(a_circle) * V(:, end);

    else
        U_1_matrix = U(:, 1:rank_A); % n*r
        Sigma_1_matrix = Sigma(1:rank_A, 1:rank_A); % r*r
        V_1_matrix = V(:, 1:rank_A); % p*r
        % so A = U_1*S_1*V_1'
        vector_singular_values = diag(Sigma_1_matrix);
        % vector, r*1, sigma_1>=...>=sigma_r>0.

        %------------------------------------------------------------------
        x_vector_OLS = hat_x_vector_lambda(rank_A, ...
            U_1_matrix, V_1_matrix, vector_singular_values, b_vector, 0);

        %------------------------------------------------------------------
        if norm(x_vector_OLS, 2) == a_circle
            x_vector_opt = x_vector_OLS;

        else
            myfun = @(lambda) function_of_lambda(rank_A, ...
                U_1_matrix, vector_singular_values, ...
                b_vector, lambda, a_circle);
            [hat_lambda, ~, EXITFLAG] = fzero(@(lambda) myfun(lambda), 0);
            if EXITFLAG ~= 1
                disp([' EXITFLAG of fzero = ', num2str(EXITFLAG)])
            end

            x_vector_opt = hat_x_vector_lambda(rank_A, ...
                U_1_matrix, V_1_matrix, vector_singular_values, ...
                b_vector, hat_lambda);
        end
    end
    %----------------------------------------------------------------------

end

end

%--------------------------------------------------------------------------
function x_vector_lambda = hat_x_vector_lambda(rank_A, ...
    U_1_matrix, V_1_matrix, vector_singular_values, b_vector, lambda)

%--------------------------------------------------------------------------
% Function: compute hat_x_vector_lambda =
% sum_{j=1}^r sigma_j (u_j_vector^T b_vector)/(sigma_j^2+\lambda) * v_j_vector
%--------------------------------------------------------------------------

x_vector_lambda = 0;
for j = 1:rank_A
    sigma_j = vector_singular_values(j);

    x_vector_lambda = x_vector_lambda + ...
        ( sigma_j * (U_1_matrix(:, j)'*b_vector) / (sigma_j^2 + lambda) ) ...
        * V_1_matrix(:, j);
end
end

%==========================================================================
function F_lambda = function_of_lambda(rank_A, ...
    U_1_matrix, vector_singular_values, b_vector, lambda, a_circle)

%--------------------------------------------------------------------------
% Function: compute
% sum_{j=1}^r [sigma_j (u_j_vector^T b_vector)/(sigma_j^2+\lambda)]^2 - a_circle^2
%--------------------------------------------------------------------------

F_lambda = 0;
for j = 1:rank_A
    sigma_j = vector_singular_values(j);

    F_lambda = F_lambda + ...
        ( sigma_j * (U_1_matrix(:, j)'*b_vector) / (sigma_j^2 + lambda) )^2;
end

F_lambda = F_lambda - a_circle^2;

end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 8 &&&&&&&&&&&&&&&&&&&&&&

function [mu_vector, cov_matrix] = mean_cov_mixture_Gaussians_D_dim...
    (prop_vector, mu_matrix, sigma_matrix)

%-------------------------------------------------------------------------
% Function: compute the (population) mean vector, cov matrix of
%           a D-dim Gaussian mixture of K-components
%-------------------------------------------------------------------------
% <Input>:
% prop_vector : = (p_1,...,p_K)': K*1 vector
% mu_matrix   : = (mu_1_vector,...,mu_K_vector), D*K matrix,
%   where mu_1_vector = (mu_11,...,mu_D1)',...
% sigma_matrix: = (sigma_1_vector,...,sigma_K_vector): D*K matrix
%-------------------------------------------------------------------------
% <Output>:
% mu_vector : D*1, vector
% cov_matrix: D*D, matrix
%-------------------------------------------------------------------------

[D, ~] = size(mu_matrix);

mu_vector = zeros(D,1);
cov_matrix = zeros(D,D);
for d = 1:D
    %-----------------------------------------------------
    mu_vector(d) = sum( prop_vector' .* mu_matrix(d,:) );

    %-----------------------------------------------------
    var_d = sum( prop_vector' .* (mu_matrix(d,:).^2 + sigma_matrix(d,:).^2) ) ...
        - (mu_vector(d))^2;
    cov_matrix(d,d) = var_d;
end

for d = 1:D
    %-----------------------------------------------------
    for k = (d+1):D
        cov_matrix(k,d) = ...
            sum( prop_vector' .* mu_matrix(k,:) .* mu_matrix(d,:) ) ...
            - mu_vector(k) * mu_vector(d);
        cov_matrix(d, k) = cov_matrix(k,d);
    end
end
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 9 &&&&&&&&&&&&&&&&&&&&&&

function [data_matrix] = mixture_Gaussians_D_dim_generator...
    (prop_vector, mu_matrix, sigma_matrix, num_obs)

%-------------------------------------------------------------------------
% Function: generate data matrix from a D-dim Gaussian mixture of K-components
%           G =
%           p_1*N( (mu_11,...,mu_D1)', (sigma_11^2,...,sigma_D1^2)' ) +...+
%           p_K*N( (mu_1K,...,mu_DK)', (sigma_1K^2,...,sigma_DK^2)' )
% Called  : discrete_label_generator.m
%-------------------------------------------------------------------------
% <Input>:
% prop_vector : = (p_1,...,p_K)': K*1 vector
% mu_matrix   : = (mu_1_vector,...,mu_K_vector), D*K matrix,
%   where mu_1_vector = (mu_11,...,mu_D1)',...
% sigma_matrix: = (sigma_1_vector,...,sigma_K_vector): D*K matrix
%   num_obs   :  n
%-------------------------------------------------------------------------
% <Output>:
% data_matrix:  = (X_1_vector,...,X_n_vector): D*num_obs, matrix
%-------------------------------------------------------------------------

[dim_G] = size(mu_matrix, 1); % dimension D of the joint distribution G

data_matrix = zeros(dim_G, num_obs);  % D*n matrix

% %-----------------------------------------------------------------------
% [K] = size(mu_matrix, 2); % num K of components in the mixture distribution G
% if dim_G == 1 && K == 2 % simple method
%     % generate a vector of data from a mixture of 2 Gaussian distributions
%     % p_1*N(mu_1, sigma_1^2) + (1-p_1)*N(mu_2, sigma_2^2)
%
%     prop_1 = prop_vector(1); % p_1
%     mu_1    = mu_matrix(1);    mu_2    = mu_matrix(2);
%     sigma_1 = sigma_matrix(1); sigma_2 = sigma_matrix(2);
%
%     U_vector = unifrnd(0, 1, num_obs, 1);  % Unif(0,1)
%     X_vector_1 = normrnd(mu_1, sigma_1, num_obs, 1);  % N(mu_1, sigma_1^2)
%     X_vector_2 = normrnd(mu_2, sigma_2, num_obs, 1);  % N(mu_2, sigma_2^2)
%
%     X_vector = X_vector_1 .* (U_vector <= prop_1) + ...
%         X_vector_2 .* (U_vector > prop_1);
%     % num_obs*1, column vector, mixture of 2 Gaussian distributions
%
%     data_matrix = X_vector'; % 1*num_obs, row vector
% %-----------------------------------------------------------------------
% else

for i = 1:num_obs
    L_rv = discrete_label_generator(prop_vector, 1); % scalar label r.v.
    Z_vector = normrnd(0, 1, dim_G, 1); % D*1 vector

    data_matrix(:, i) = mu_matrix(:, L_rv) + ...
        sigma_matrix(:, L_rv) .* Z_vector; % D*num_obs, matrix
end
end


%&&&&&&&&&&&&&&&&&&&&&& scripts for code 10 &&&&&&&&&&&&&&&&&&&&&&

function pdf_val = pdf_mixture_Gaussians_D_dim_func...
    (prop_vector, mu_matrix, sigma_matrix, x_vector)

%-------------------------------------------------------------------------
% Function: compute the pdf of a D-dim Gaussian mixture of K-components
%-------------------------------------------------------------------------
% <Input>:
% prop_vector : = (p_1,...,p_K)': K*1 vector
% mu_matrix   : = (mu_1_vector,...,mu_K_vector), D*K matrix,
%   where mu_1_vector = (mu_11,...,mu_D1)',...
% sigma_matrix: = (sigma_1_vector,...,sigma_K_vector): D*K matrix
%  x_vector   : D*1 vector, (x_1,...,x_D)'
%-------------------------------------------------------------------------
% <Output>:
% pdf_val: scalar
%-------------------------------------------------------------------------

[D, K] = size(mu_matrix);

f_vector = zeros(K, 1); % f(1),...,f(K)
phi_vector = zeros(D, 1); % phi(1),...,phi(D)
for k = 1:K
    for d = 1:D
        x_d = x_vector(d);
        mu_d_k = mu_matrix(d, k);
        sigma_d_k = sigma_matrix(d, k);

        phi_vector(d) = normpdf( (x_d-mu_d_k)/sigma_d_k, 0, 1 )/sigma_d_k;
    end

    f_vector(k) = prod(phi_vector);
end

pdf_val = sum(prop_vector .* f_vector); % scalar
end


%&&&&&&&&&&&&&&&&&&&&&& scripts for code 11 &&&&&&&&&&&&&&&&&&&&&&

function [X_matrix] = X_matrix_sim(choice_F, dim_p, num_obs, mu_F, A, B)

% choice_F: Gaussian, Uniforma, Beta
% dim_p   : dimension
% num_obs : # of data points
%  mu_F   : expectation of F
%   A     : parameter 1 in F
%   B     : parameter 2 in F
%----------------------------------------
% X_matrix:  % p*n matrix

%--------------------- generate data matrix ---------------------------
if     choice_F == 0
    X_matrix = normrnd(A, B, dim_p, num_obs) - mu_F; % mu_F_centered = 0

elseif choice_F == 1
    X_matrix = unifrnd(A, B, dim_p, num_obs) - mu_F; % mu_F_centered = 0

elseif choice_F == 2
    X_matrix = 6 * (betarnd(A, B, dim_p, num_obs) - mu_F); % mu_F_centered = 0
end
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 12 &&&&&&&&&&&&&&&&&&&&&&

function [z_vector] = z_vector_exist_dep_on_X_G(X_matrix, a_vector_G, opt_display)

%--------------------------------------------------------------------------
% Function: compute the direction vector (in the "unit sphere") that
%           "exists" in the feasibility results in BKN (2018), where p >= n
%--------------------------------------------------------------------------
% <Input>:
% X_matrix: p*n matrix; X'X is n*n, we assume p >= n,
%           so X'X is invertible.
% a_vector_G: n*1 vector,
% opt_display: 0 for "without" display; 1 for "with" display
%--------------------------------------------------------------------------
% <Output>:
% z_vector: p*1 vector, requiring \|z_vector\|_2 = 1
%--------------------------------------------------------------------------

[dim_p, num_obs] = size(X_matrix);
if dim_p < num_obs
    disp(' dim_p < num_obs; return!!!'); % we assume p >= n
    return
end

%====================== obtain z_vector_0 =================================
z_vector_0 = X_matrix / (X_matrix' * X_matrix) * a_vector_G;
norm_squared_z_vector_0 = sum(z_vector_0.^2); % \|z_vector_0\|_2^2
if opt_display == 1
    disp([' \|z_vector_0\|_2^2 = ', num2str(norm_squared_z_vector_0)])
end

%====================== modify z_vector_0 =================================
if     dim_p == num_obs
    if norm_squared_z_vector_0 == 1
        z_vector = z_vector_0; % p*1 vector, \|z_vector\|_2 = 1
    else
        disp(' norm_squared_z_vector_0 \ne 1; return!!!')
        return
    end

elseif dim_p > num_obs
    [~, ~, V_matrix] = svd(X_matrix');
    %     [U,S,V] = svd(X) produces a diagonal matrix S, of the same
    %     dimension as X and with nonnegative diagonal elements in
    %     decreasing order, and unitary matrices U and V so that X = U*S*V'.
    % So, here, X_matrix' = U S V^T, where X_matrix' is n*p,
    % U is n*n, S is n*p, V is p*p.

    z_vector = z_vector_0 + ...
        sqrt( 1 - norm_squared_z_vector_0 ) * V_matrix(:, (num_obs+1));
    % p*1 vector, \|z_vector\|_2 = 1
end
end

%&&&&&&&&&&&&&&&&&&&&&& scripts for code 13 &&&&&&&&&&&&&&&&&&&&&&

function [z_vector] = z_vector_unit_Euclidean_norm( X_vector )

% Function: transfer X_vector with unit Euclidean norm
%--------------------------------------------------------------------------
% <Input>:
% X_matrix: vector
%--------------------------------------------------------------------------
% <Output>:
% z_vector: vector, X_vector / |X_vector|_2
%--------------------------------------------------------------------------

z_vector = X_vector / ( sum(X_vector.^2)^(1/2) );
% vector, on the unit sphere

end
