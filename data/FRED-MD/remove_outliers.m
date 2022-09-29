function [Y,n]=remove_outliers(X)
% =========================================================================
% DESCRIPTION:
% This function takes a set of series aligned in the columns of a matrix
% and replaces outliers with the value NaN.
%
% -------------------------------------------------------------------------
% INPUT:
%           X   = dataset (one series per column)
% 
% OUTPUT:
%           Y   = dataset with outliers replaced with NaN 
%           n   = number of outliers found in each series
%
% -------------------------------------------------------------------------
% NOTES:
%           1) Outlier definition: a data point x of a series X(:,i) is
%           considered an outlier if abs(x-median)>10*interquartile_range.
%
%           2) This function ignores values of NaN and thus is capable of
%           replacing outliers for series that have missing values.
%
% =========================================================================
% FUNCTION:

% Calcualte median of each series
median_X=nanmedian(X,1);

% Repeat median of each series over all data points in the series
median_X_mat=repmat(median_X,size(X,1),1);

% Calculate quartiles 
Q=prctile(X,[25, 50, 75],1);

% Calculate interquartile range (IQR) of each series
IQR=Q(3,:)-Q(1,:);

% Repeat IQR of each series over all data points in the series
IQR_mat=repmat(IQR,size(X,1),1);

% Determine outliers 
Z=abs(X-median_X_mat);
outlier=Z>(10*IQR_mat);

% Replace outliers with NaN
Y=X;
Y(outlier)=NaN;

% Count number of outliers
n=sum(outlier,1);
n

end

