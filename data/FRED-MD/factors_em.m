function [ehat,Fhat,lamhat,ve2,x2] = factors_em(x,kmax,jj,DEMEAN)
% =========================================================================
% DESCRIPTION
% This program estimates a set of factors for a given dataset using
% principal component analysis. The number of factors estimated is
% determined by an information criterion specified by the user. Missing
% values in the original dataset are handled using an iterative
% expectation-maximization (EM) algorithm.
%
% -------------------------------------------------------------------------
% INPUTS
%           x       = dataset (one series per column)
%           kmax    = an integer indicating the maximum number of factors
%                     to be estimated; if set to 99, the number of factors
%                     selected is forced to equal 8
%           jj      = an integer indicating the information criterion used 
%                     for selecting the number of factors; it can take on 
%                     the following values:
%                           1 (information criterion PC_p1)
%                           2 (information criterion PC_p2)
%                           3 (information criterion PC_p3)      
%           DEMEAN  = an integer indicating the type of transformation
%                     performed on each series in x before the factors are
%                     estimated; it can take on the following values:
%                           0 (no transformation)
%                           1 (demean only)
%                           2 (demean and standardize)
%                           3 (recursively demean and then standardize) 
%
% OUTPUTS
%           ehat    = difference between x and values of x predicted by
%                     the factors
%           Fhat    = set of factors
%           lamhat  = factor loadings
%           ve2     = eigenvalues of x3'*x3 (where x3 is the dataset x post
%                     transformation and with missing values filled in)
%           x2      = x with missing values replaced from the EM algorithm
%
% -------------------------------------------------------------------------
% SUBFUNCTIONS
%
% baing() - selects number of factors
%
% pc2() - runs principal component analysis
%
% minindc() - finds the index of the minimum value for each column of a
%       given matrix
%
% transform_data() - performs data transformation
%
% -------------------------------------------------------------------------
% BREAKDOWN OF THE FUNCTION
%
% Part 1: Check that inputs are specified correctly.
%
% Part 2: Setup.
%
% Part 3: Initialize the EM algorithm -- fill in missing values with
%         unconditional mean and estimate factors using the updated
%         dataset.
%
% Part 4: Perform the EM algorithm -- update missing values using factors,
%         construct a new set of factors from the updated dataset, and
%         repeat until the factor estimates do not change.
% 
% -------------------------------------------------------------------------
% NOTES
% Authors: Michael W. McCracken and Serena Ng
% Date: 9/5/2017
% Version: MATLAB 2014a
% Required Toolboxes: None
%
% Details for the three possible information criteria can be found in the
% paper "Determining the Number of Factors in Approximate Factor Models" by
% Bai and Ng (2002).
%
% The EM algorithm is essentially the one given in the paper "Macroeconomic
% Forecasting Using Diffusion Indexes" by Stock and Watson (2002). The
% algorithm is initialized by filling in missing values with the
% unconditional mean of the series, demeaning and standardizing the updated
% dataset, estimating factors from this demeaned and standardized dataset,
% and then using these factors to predict the dataset. The algorithm then
% proceeds as follows: update missing values using values predicted by the
% latest set of factors, demean and standardize the updated dataset,
% estimate a new set of factors using the demeaned and standardized updated
% dataset, and repeat the process until the factor estimates do not change.
%
% =========================================================================
% PART 1: CHECKS

% Check that x is not missing values for an entire row
if sum(sum(isnan(x),2)==size(x,2))>0
    error('Input x contains entire row of missing values.');
end

% Check that x is not missing values for an entire column
if sum(sum(isnan(x),1)==size(x,1))>0
    error('Input x contains entire column of missing values.');
end

% Check that kmax is an integer between 1 and the number of columns of x,
% or 99
if ~((kmax<=size(x,2) && kmax>=1 && floor(kmax)==kmax) || kmax==99)
    error('Input kmax is specified incorrectly.');
end

% Check that jj is one of 1, 2, 3
if jj~=1 && jj~=2 && jj~=3
    error('Input jj is specified incorrectly.');
end

% Check that DEMEAN is one of 0, 1, 2, 3
if DEMEAN ~= 0 && DEMEAN ~= 1 && DEMEAN ~= 2 && DEMEAN ~= 3
    error('Input DEMEAN is specified incorrectly.');
end

% =========================================================================
% PART 2: SETUP

% Maximum number of iterations for the EM algorithm
maxit=50;

% Number of observations per series in x (i.e. number of rows)
T=size(x,1);

% Number of series in x (i.e. number of columns)
N=size(x,2);

% Set error to arbitrarily high number
err=999;

% Set iteration counter to 0
it=0;

% Locate missing values in x
x1=isnan(x);

% =========================================================================
% PART 3: INITIALIZE EM ALGORITHM
% Fill in missing values for each series with the unconditional mean of
% that series. Demean and standardize the updated dataset. Estimate factors
% using the demeaned and standardized dataset, and use these factors to
% predict the original dataset.

% Get unconditional mean of the non-missing values of each series
mut=repmat(nanmean(x),T,1);

% Replace missing values with unconditional mean
x2=x;
x2(isnan(x))=mut(isnan(x));

% Demean and standardize data using subfunction transform_data()
%   x3  = transformed dataset
%   mut = matrix containing the values subtracted from x2 during the
%         transformation
%   sdt = matrix containing the values that x2 was divided by during the
%         transformation
[x3,mut,sdt]=transform_data(x2,DEMEAN);

% If input 'kmax' is not set to 99, use subfunction baing() to determine
% the number of factors to estimate. Otherwise, set number of factors equal
% to 8
if kmax ~=99
    [icstar,~,~,~]=baing(x3,kmax,jj);
else
    icstar=8;
end

% Run principal components on updated dataset using subfunction pc2()
%   chat   = values of x3 predicted by the factors
%   Fhat   = factors scaled by (1/sqrt(N)) where N is the number of series
%   lamhat = factor loadings scaled by number of series
%   ve2    = eigenvalues of x3'*x3 
[chat,Fhat,lamhat,ve2]  = pc2(x3,icstar);

% Save predicted series values
chat0=chat;

% =========================================================================
% PART 4: PERFORM EM ALGORITHM
% Update missing values using values predicted by the latest set of
% factors. Demean and standardize the updated dataset. Estimate a new set
% of factors using the updated dataset. Repeat the process until the factor
% estimates do not change.

% Run while error is large and have yet to exceed maximum number of
% iterations
while err> 0.000001 && it <maxit
    
    % ---------------------------------------------------------------------
    % INCREASE ITERATION COUNTER
    
    % Increase iteration counter by 1
    it=it+1;
    
    % Display iteration counter, error, and number of factors
    fprintf('Iteration %d: obj %10f IC %d \n',it,err,icstar);

    % ---------------------------------------------------------------------
    % UPDATE MISSING VALUES
    
    % Replace missing observations with latest values predicted by the
    % factors (after undoing any transformation)
    for t=1:T;
        for j=1:N;
            if x1(t,j)==1
                x2(t,j)=chat(t,j)*sdt(t,j)+mut(t,j);    
            else
                x2(t,j)=x(t,j);
            end
        end
    end
    
    % ---------------------------------------------------------------------
    % ESTIMATE FACTORS
    
    % Demean/standardize new dataset and recalculate mut and sdt using
    % subfunction transform_data()
    %   x3  = transformed dataset
    %   mut = matrix containing the values subtracted from x2 during the
    %         transformation
    %   sdt = matrix containing the values that x2 was divided by during 
    %         the transformation
    [x3,mut,sdt]=transform_data(x2,DEMEAN);
        
    % Determine number of factors to estimate for the new dataset using
    % subfunction baing() (or set to 8 if kmax equals 99)
    if kmax ~=99
        [icstar,~,~,~]=baing(x3,kmax,jj);
    else
        icstar=8;
    end

    % Run principal components on the new dataset using subfunction pc2()
    %   chat   = values of x3 predicted by the factors
    %   Fhat   = factors scaled by (1/sqrt(N)) where N is the number of 
    %            series
    %   lamhat = factor loadings scaled by number of series
    %   ve2    = eigenvalues of x3'*x3 
    [chat,Fhat,lamhat,ve2]  = pc2(x3,icstar);

    % ---------------------------------------------------------------------
    % CALCULATE NEW ERROR VALUE
    
    % Caclulate difference between the predicted values of the new dataset
    % and the predicted values of the previous dataset
    diff=chat-chat0;
    
    % The error value is equal to the sum of the squared differences
    % between chat and chat0 divided by the sum of the squared values of
    % chat0
    v1=diff(:);
    v2=chat0(:);
    err=(v1'*v1)/(v2'*v2);

    % Set chat0 equal to the current chat
    chat0=chat;
end

% Produce warning if maximum number of iterations is reached
if it==maxit
    warning('Maximum number of iterations reached in EM algorithm');
end

% -------------------------------------------------------------------------
% FINAL DIFFERNECE

% Calculate the difference between the initial dataset and the values 
% predicted by the final set of factors
ehat = x-chat.*sdt-mut;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         SUBFUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ic1,chat,Fhat,eigval]=baing(X,kmax,jj)
% =========================================================================
% DESCRIPTION
% This function determines the number of factors to be selected for a given
% dataset using one of three information criteria specified by the user.
% The user also specifies the maximum number of factors to be selected.
%
% -------------------------------------------------------------------------
% INPUTS
%           X       = dataset (one series per column)
%           kmax    = an integer indicating the maximum number of factors
%                     to be estimated
%           jj      = an integer indicating the information criterion used 
%                     for selecting the number of factors; it can take on 
%                     the following values:
%                           1 (information criterion PC_p1)
%                           2 (information criterion PC_p2)
%                           3 (information criterion PC_p3)    
%
% OUTPUTS
%           ic1     = number of factors selected
%           chat    = values of X predicted by the factors
%           Fhat    = factors
%           eigval  = eivenvalues of X'*X (or X*X' if N>T)
%
% -------------------------------------------------------------------------
% SUBFUNCTIONS USED
%
% minindc() - finds the index of the minimum value for each column of a
%       given matrix
%
% -------------------------------------------------------------------------
% BREAKDOWN OF THE FUNCTION
%
% Part 1: Setup.
%
% Part 2: Calculate the overfitting penalty for each possible number of
%         factors to be selected (from 1 to kmax).
%
% Part 3: Select the number of factors that minimizes the specified
%         information criterion by utilizing the overfitting penalties
%         calculated in Part 2.
%
% Part 4: Save other output variables to be returned by the function (chat,
%         Fhat, and eigval). 
%
% =========================================================================
% PART 1: SETUP

% Number of observations per series (i.e. number of rows)
T=size(X,1);

% Number of series (i.e. number of columns)
N=size(X,2);

% Total number of observations
NT=N*T;

% Number of rows + columns
NT1=N+T;

% =========================================================================
% PART 2: OVERFITTING PENALTY
% Determine penalty for overfitting based on the selected information
% criterion. 

% Allocate memory for overfitting penalty
CT=zeros(1,kmax);

% Array containing possible number of factors that can be selected (1 to
% kmax)
ii=1:1:kmax;

% The smaller of N and T
GCT=min([N;T]);

% Calculate penalty based on criterion determined by jj. 
switch jj
    
    % Criterion PC_p1
    case 1
        CT(1,:)=log(NT/NT1)*ii*NT1/NT;
        
    % Criterion PC_p2
    case 2
        CT(1,:)=(NT1/NT)*log(min([N;T]))*ii;
        
    % Criterion PC_p3
    case 3
        CT(1,:)=ii*log(GCT)/GCT;
        
end

% =========================================================================
% PART 3: SELECT NUMBER OF FACTORS
% Perform principal component analysis on the dataset and select the number
% of factors that minimizes the specified information criterion.

% -------------------------------------------------------------------------
% RUN PRINCIPAL COMPONENT ANALYSIS

% Get components, loadings, and eigenvalues
if T<N 
    
    % Singular value decomposition
    [ev,eigval,~]=svd(X*X'); 
    
    % Components
    Fhat0=sqrt(T)*ev;
    
    % Loadings
    Lambda0=X'*Fhat0/T;
    
else
    
    % Singular value decomposition
    [ev,eigval,~]=svd(X'*X);
    
    % Loadings
    Lambda0=sqrt(N)*ev;
    
    % Components
    Fhat0=X*Lambda0/N;

end

% -------------------------------------------------------------------------
% SELECT NUMBER OF FACTORS 
    
% Preallocate memory
Sigma=zeros(1,kmax+1); % sum of squared residuals divided by NT
IC1=zeros(size(CT,1),kmax+1); % information criterion value

% Loop through all possibilites for the number of factors 
for i=kmax:-1:1

    % Identify factors as first i components
    Fhat=Fhat0(:,1:i);

    % Identify factor loadings as first i loadings
    lambda=Lambda0(:,1:i);

    % Predict X using i factors
    chat=Fhat*lambda';

    % Residuals from predicting X using the factors
    ehat=X-chat;

    % Sum of squared residuals divided by NT
    Sigma(i)=mean(sum(ehat.*ehat/T));

    % Value of the information criterion when using i factors
    IC1(:,i)=log(Sigma(i))+CT(:,i);
    
end

% Sum of squared residuals when using no factors to predict X (i.e.
% fitted values are set to 0)
Sigma(kmax+1)=mean(sum(X.*X/T));

% Value of the information criterion when using no factors
IC1(:,kmax+1)=log(Sigma(kmax+1));

% Number of factors that minimizes the information criterion
ic1=minindc(IC1')';

% Set ic1=0 if ic1>kmax (i.e. no factors are selected if the value of the
% information criterion is minimized when no factors are used)
ic1=ic1 .*(ic1 <= kmax);

% =========================================================================
% PART 4: SAVE OTHER OUTPUT

% Factors and loadings when number of factors set to kmax
Fhat=Fhat0(:,1:kmax); % factors
Lambda=Lambda0(:,1:kmax); % factor loadings

% Predict X using kmax factors
chat=Fhat*Lambda';

% Get the eivenvalues corresponding to X'*X (or X*X' if N>T)
eigval=diag(eigval);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [chat,fhat,lambda,ss]=pc2(X,nfac)
% =========================================================================
% DESCRIPTION
% This function runs principal component analysis.
%
% -------------------------------------------------------------------------
% INPUTS
%           X      = dataset (one series per column)
%           nfac   = number of factors to be selected
%
% OUTPUTS
%           chat   = values of X predicted by the factors
%           fhat   = factors scaled by (1/sqrt(N)) where N is the number of
%                    series
%           lambda = factor loadings scaled by number of series
%           ss     = eigenvalues of X'*X 
%
% =========================================================================
% FUNCTION

% Number of series in X (i.e. number of columns)
N=size(X,2);

% Singular value decomposition: X'*X = U*S*V'
[U,S,~]=svd(X'*X);

% Factor loadings scaled by sqrt(N)
lambda=U(:,1:nfac)*sqrt(N);

% Factors scaled by 1/sqrt(N) (note that lambda is scaled by sqrt(N))
fhat=X*lambda/N;

% Estimate initial dataset X using the factors (note that U'=inv(U))
chat=fhat*lambda';

% Identify eigenvalues of X'*X
ss=diag(S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function pos=minindc(x)
% =========================================================================
% DESCRIPTION
% This function finds the index of the minimum value for each column of a
% given matrix. The function assumes that the minimum value of each column
% occurs only once within that column. The function returns an error if
% this is not the case.
%
% -------------------------------------------------------------------------
% INPUT
%           x   = matrix 
%
% OUTPUT
%           pos = column vector with pos(i) containing the row number
%                 corresponding to the minimum value of x(:,i)
%
% =========================================================================
% FUNCTION

% Number of rows and columns of x
nrows=size(x,1);
ncols=size(x,2);

% Preallocate memory for output array
pos=zeros(ncols,1);

% Create column vector 1:nrows
seq=(1:nrows)';

% Find the index of the minimum value of each column in x
for i=1:ncols
    
    % Minimum value of column i
    min_i=min(x(:,i));
    
    % Column vector containing the row number corresponding to the minimum
    % value of x(:,i) in that row and zeros elsewhere
    colmin_i= seq.*((x(:,i)-min_i)==0);
    
    % Produce an error if the minimum value occurs more than once
    if sum(colmin_i>0)>1
        error('Minimum value occurs more than once.');
    end
    
    % Obtain the index of the minimum value by taking the sum of column
    % vector 'colmin_i'
    pos(i)=sum(colmin_i);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [x22,mut,sdt]=transform_data(x2,DEMEAN)
% =========================================================================
% DESCRIPTION
% This function transforms a given set of series based upon the input
% variable DEMEAN. The following transformations are possible:
%
%   1) No transformation.
%   
%   2) Each series is demeaned only (i.e. each series is rescaled to have a
%   mean of 0).
%   
%   3) Each series is demeaned and standardized (i.e. each series is
%   rescaled to have a mean of 0 and a standard deviation of 1).
%   
%   4) Each series is recursively demeaned and then standardized. For a
%   given series x(t), where t=1,...,T, the recursively demeaned series
%   x'(t) is calculated as x'(t) = x(t) - mean(x(1:t)). After the
%   recursively demeaned series x'(t) is calculated, it is standardized by
%   dividing x'(t) by the standard deviation of the original series x. Note
%   that this transformation does not rescale the original series to have a
%   specified mean or standard deviation.
%
% -------------------------------------------------------------------------
% INPUTS
%           x2      = set of series to be transformed (one series per
%                     column); no missing values;
%           DEMEAN  = an integer indicating the type of transformation
%                     performed on each series in x2; it can take on the
%                     following values:
%                           0 (no transformation)
%                           1 (demean only)
%                           2 (demean and standardize)
%                           3 (recursively demean and then standardize) 
%
% OUTPUTS
%           x22     = transformed dataset
%           mut     = matrix containing the values subtracted from x2
%                     during the transformation
%           sdt     = matrix containing the values that x2 was divided by
%                     during the transformation
%
% =========================================================================
% FUNCTION

% Number of observations in each series (i.e. number of rows in x2)
T=size(x2,1);

% Number of series (i.e. number of columns in x2)
N=size(x2,2);

% Perform transformation based on type determined by 'DEMEAN'
switch DEMEAN
    
    % ---------------------------------------------------------------------
    % No transformation
    case 0
        mut=repmat(zeros(1,N),T,1);
        sdt=repmat(ones(1,N),T,1);
        x22=x2;
        
    % ---------------------------------------------------------------------
    % Each series is demeaned only
    case 1
        mut=repmat(mean(x2),T,1);
        sdt=repmat(ones(1,N),T,1);
        x22=x2-mut;
        
    % ---------------------------------------------------------------------
    % Each series is demeaned and standardized 
    case 2
        mut=repmat(mean(x2),T,1);
        sdt=repmat(std(x2),T,1);
        x22=(x2-mut)./sdt;
        
    % ---------------------------------------------------------------------
    % Each series is recursively demeaned and then standardized
    case 3
        mut=NaN(size(x2));
        for t=1:T
            mut(t,:)=mean(x2(1:t,:),1);
        end
        sdt=repmat(std(x2),T,1);
        x22=(x2-mut)./sdt; 
end





