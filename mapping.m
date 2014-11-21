function mapping = create_word_map(sentences)
    map = containers.Map();

    
    
    for i = 1:length(sentences)
        word_list = strsplit(sentences{i}, ' ');
        map(i) = zeros(length(word_list), 1);
        
        for i = 1:length(word_list)
            word = 
            