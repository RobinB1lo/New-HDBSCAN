From now on we will keep our questions and meeting notes in this directory. THe stage we are currently at is the implementation of deconvolution in the algorithm. 

## Meeting 1 (since starting this dir)

### Progress

- Since last meeting we have not made much progress since we both had exams during the week, howveer we were able to implement the change of variable formula to a certain extent (we are stuck at re verting our observations)

### Questions

- For change of variable process, how do we change back the variables after doing the kernel denstity estimations for the 1D estimates. I have read about the Jacobian correction factor, is that what you think we should do?

- Would it be possible to re-explain the formula we spoke about last meeting, I did not fully understand when looking back at the picture and therefore I was not able to implement it?

- When you said we should look in hdbscan when the probability is determined using counting, should we strictly look in the boruvka.py file? (where the sample_weights_core_distances function is found)
    - Look through all the files 

- We still have some way to go for this project, but I was reading up on SAE vizualiations which was in your initial email and I was wondering if you think we could make any contributions to that project. If you think it is best to simply stick with New-HDBSCAN I completely understand also, but we do not mind at all to be working on both projects as they are both super interesting
    - If yes, could we have an intro to the project 

### Notes during meeting

- No nice deconvolution fomrula for squared observations (so it makes change of variable difficult)

- We should start wit implementing the posterior probability formula that we spoke about last time (the one inside the algorithm)

    - The formula is in the picture, 1 is inputtable into vanilla hdbscan, and 2 is the same inputting into monte carlo but different computation
        - Try in 2d and to get a good estimate 
        - Maybe mess with the algebra for the distances computatoon (Read this paper it could help ("Tykanov Regularization"))

- Read the algorithm line by line and see if you ever think ("i can view this as an estimate") and where you can improve this

- Pre print on SAE vizualtion (David Blei most recent papers on topic modelling) (look up on arxive) (enjalot blog and github for sae viz)

- Vizualitons in google coolab (you should do the vizualitions of enjalot's github in google coolab)

- Reserach what "Topic modelling" is 

- Reserach "Bert Topic document retreival" (the most basic document summary and fetching algorithm)

- Splade algo for doc retrieval

- SAE are autoencoders, 

- RESEARCH AUTOUENCODERS BEFORE ANYTHING ON SAE 

- You take the in between of the LLM process, and train with that data, you then 

- Look at Golden gate paper for llms (the llm likes the golden gate bridge)

- Word to vector research paper (famous millions of citations) (with the golden gate bridge example the vector v = golden gate bridge)

- SAE improves the vectors (somehow....How???) (sometimes works, fails all the time tho)

- Empirical word meaning

- 