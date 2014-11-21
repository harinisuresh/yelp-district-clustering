%documents = list of documents in cell array
%alpha = vector of length number_of_topics i.e. [0.1, 0.1, 0.1......0.1]
%beta = vector of length N [0.0001, 0.0001, 0.0001.........0.0001] prob of
%each word w getting assigned to any topic 

function output = lda_gibbs_sampler(number_of_topics, number_of_iterations, words, document_numbers, alpha, beta)
    w = words;  
    d = document_numbers;
    N = length(w);
    z = zeros(N, 1); 
    n_d_k = containers.Map(); %document -> list of topics
    n_k = zeros(number_of_topics, 1); %list of topics
    n_k_w = containers.Map(); %word -> list of topics
    
    for i = 1:N
        z(i) = randi([1,number_of_topics]);
    end
    
    for m = 1:number_of_iterations
        for i = 1:N 
            word = w(i);
            topic = z(i);
            doc_number = d(i);
            current_doc_topic_list = n_d_k(doc_number);
            current_doc_topic_list(topic) = current_doc_topic_list(topic) - 1;
            current_word_topic_list = n_k_w(i);
            current_word_topic_list(topic) = current_word_topic_list(topic) - 1;
            current_topic_word_count = n_k(topic);
            n_k(topic) = current_topic_word_count - 1;
            for j = 1:number_of_topics
                n_d_topic = current_doc_topic_list(j);
                alpha_k = alpha(j);
                n_topic_w = current_word_topic_list(j);
                beta_w = beta(i);
                n_topic = n_k(j);
                p = zeros(number_of_topics);
                p(j) = (n_d_topic + alpha_k)*(n_topic_w + beta_w)/(n_topic + beta*w); %need to add somethign to w so it only considers the first column
           
            end
            p = p/sum(p);
            total = 0;
            random = rand(1);
            for n = 1:number_of_topics
                total = total + p(n);
                if random <= total
                    sample = n;
                    break
                end
            end
            z(i) = sample;
            current_doc_topic_list(sample) = current_doc_topic_list(sample) + 1;
            current_word_topic_list(sample) = current_word_topic_list(sample) + 1;
            n_k(sample) = n_k(sample) + 1;
        end
    end
    
    output = [z, n_d_k, n_k_w, n_k];
          
end            
            