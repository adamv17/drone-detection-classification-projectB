   
classdef comp_functions
    methods (Static)
        function [score] = mean_PSD_grading(className, psd_to_compare, dataBase) 
           
            allPSDs = [dataBase.(className).inst.PSD]; 
            meanPSD = mean(allPSDs, 2); 
            score = sum(psd_to_compare .* meanPSD);
        end 
        

        function [max_score] = individual_PSD_grading(className, psd_to_compare, dataBase) 
        numInstances = numel(dataBase.(className).inst);
        scores_arr = zeros(1, numInstances);
        for i = 1:numInstances
            current_PSD = dataBase.(className).inst(i).PSD;
            scores_arr(i) = sum(psd_to_compare .* current_PSD);
        end
        
        max_score = max(scores_arr);
        
        end 
        

        function [score] = individual_mfcc_grading(className, mfcc_to_compare, dataBase) 
            numInstances = numel(dataBase.(className).inst);
            dist_arr = zeros(1, numInstances);
            
            for i = 1:numInstances
                curr_mfcc = dataBase.(className).inst(i).MFCC;
                dist_arr(i) = norm(curr_mfcc - mfcc_to_compare);
            end
            
            score = exp(-min(dist_arr) / 20); 
        end

   
    
        function [score] = prony_grading(className, prony_freq_arr_to_compare, dataBase)
            
            all_prony_freqs = [dataBase.(className).inst.Prony_freq]; 
            num_instances = size(all_prony_freqs, 2); 
               
            smallest_distance = inf;
            
            
            for i = 1:num_instances
                
                current_distance = norm(prony_freq_arr_to_compare - all_prony_freqs(:, i));
                if current_distance < smallest_distance
                    smallest_distance = current_distance;
                end
            end
            
            score = exp(-smallest_distance / 2000);
        end

    end

end

