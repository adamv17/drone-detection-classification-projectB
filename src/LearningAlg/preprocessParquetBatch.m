function [X, Y] = preprocessParquetBatch(data)
    % This function runs on every "chunk" of data read from the disk
    % Input: 'data' is a Table with columns 'audio' and 'label'
    
    numRows = height(data);
    
    % Pre-allocate output. 
    % For Deep Learning, X is usually a cell array of {Signal} or {Spectrogram}
    X = cell(numRows, 1);
    
    % Y should be categorical for classification
    Y = categorical(data.label); 
    
    for i = 1:numRows
        % 1. Extract bytes from the nested struct
        % Note: Adjust '.bytes' if your struct field is named differently
        rawBytes = data.audio.bytes{i}; 
        
        % 2. Create a unique temp filename (threaded-safe)
        tempName = [tempname '.wav']; 
        
        % 3. Write bytes to disk
        fid = fopen(tempName, 'W');
        fwrite(fid, rawBytes, 'uint8');
        fclose(fid);
        
        % 4. Read Audio (Decode)
        [sig, fs] = audioread(tempName);   
        X{i} = sig;
        
        % 5. Delete temp file
        delete(tempName);
    end
end