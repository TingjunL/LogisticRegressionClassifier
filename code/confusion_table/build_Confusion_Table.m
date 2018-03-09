% This function helps to buile the confusion table.

function [TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class)
    pred_T = pred_class == 1;
    pred_F = pred_class == 0;
    actual_T = actual_class == 1;
    actual_F = actual_class == 0;
    TP = sum(and(pred_T==actual_T,pred_T));
    TN = sum(and(pred_F==actual_F,pred_F));
    FP = sum(and(pred_T==actual_F,pred_T));
    FN = sum(and(pred_F==actual_T,pred_F));
    
    
end