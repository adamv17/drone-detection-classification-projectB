   
classdef comp_functions
    methods (Static)
        function [score] = mean_PSD_grading(className, psd_to_compare, dataBase) 
            % גישה דינמית לשדה באמצעות סוגריים: dataBase.(className)
            % שרשור כל ה-PSD למטריצה כדי לחשב ממוצע
            allPSDs = [dataBase.(className).inst.PSD]; 
            meanPSD = mean(allPSDs, 2); 
            score = sum(psd_to_compare .* meanPSD);
        end 
        
        function [score] = mean_Time_grading(className, time_sig_to_compare, dataBase) 
            % תיקון typo מ-calss ל-className ושימוש בשדה Signal
            allSignals = [dataBase.(className).inst.Signal]; 
            meanTimeSig = mean(allSignals, 2);
            score = sum(time_sig_to_compare .* meanTimeSig);
        end 
        
        function [max_score] = individual_PSD_grading(className, psd_to_compare, dataBase) 
            % numel צריך לעבוד על מערך המופעים (inst)
            numInstances = numel(dataBase.(className).inst);
            scores_arr = zeros(1, numInstances);
            for i = 1:numInstances
                current_PSD = dataBase.(className).inst(i).PSD;
                % ב-MATLAB אין פונקציית insert למערכים, משתמשים באינדקס
                scores_arr(i) = sum(psd_to_compare .* current_PSD);
            end
            max_score=max(scores_arr);
        end 
        
        function [max_score] = individual_Time_grading(className, timeSig_to_compare, dataBase) 
            numInstances = numel(dataBase.(className).inst);
            scores_arr = zeros(1, numInstances);
            for i = 1:numInstances
                % שימוש בשם השדה הנכון Signal (במקום time)
                current_timeSig = dataBase.(className).inst(i).Signal;
                % תיקון המשתמש מ-psd_to_compare ל-timeSig_to_compare
                scores_arr(i) = sum(timeSig_to_compare .* current_timeSig);
            end
            max_score=max(scores_arr);
        end
    end

end