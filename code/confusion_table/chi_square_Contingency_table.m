% This function provides additional imformation about the confusion table.

function [chi_square,sensitivity,specificity,precision] = chi_square_Contingency_table(TP,FP,FN,TN)
    N = TP+FP+FN+TN;
    
    chi_square = (TP*TN-FP*FN)^2*N/((TP+FP)*(FN+TN)*(TP+FN)*(FP+TN));
    sensitivity = TP/(TP+FN);
    specificity = TN/(TN+FP);
    precision   = TP/(TP+FP);



end