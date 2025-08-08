## To do
- Set up a virtual environment for this specific project - ✅
- Make datasets actually noisy, and do tests with increased noise and blobs - ✅ 
- Look at notes you took during meeting with Professor Smith - ✅ 
- Do step 0 that professor smith mentioned - ✅ 
- Try out step 0 with some tests - ✅ 
- What exactly are mixture models? - ✅ 
- Read and understand deconvolution - ✅ 
- Implement deconvolution (Step 1) - ✅ 
- Test Step 1 - ✅ (Did not do well)
- Implemented The "bad part" from email - ✅ 
- Read and understand second part of email - ✅
- Build an h valuye that adapts to noise -  ✅
- Look in the hdbscan code to see ways of imporving the evaluation of probability using counting and density, make a list of where each instance of this is  - ✅ 
- Implement the method mentioned last meeting - 
- Read last part of email referring to "nearest neighbours graph" - 
- The current deconvolution algorthm is outperforming hdbscan, try and improve it as much as possible before implementing the change of variable formula 
- Implement the change of variable formula to "push forward" your density into the 1 d array to then make it easier to "deconvolve"
- Improve the densisty and probability estimates from your list - 
- Answer the question "how do I actually do deconvolution when I only have access to the nearest-neighbour graph?" - 


### something we have not yet explored is changing the type of noise added to the blobs, as of right now we have only used gaussian noise. Also we have not changed hthe # of k nearest neighbours

## List of tests to try:
1. Non-convex / irregular shapes
    - Two interleaving half moons
    - Concentric circles
    - Spiral or S-curves 

Industry use: fault‑pattern detection in vibrations, geospatial zones that wrap around obstacles, customer journeys with looping behavior.

2. Varying densities
    - Gaussian blobs with different variances 
    - Clusters plus uniform "background clutter"

Industry use: sensor‐network hotspots sitting on top of broad environmental noise; anomaly points scattered in IoT telemetry.

3. Anistropic / elongated clusters
    - Spherical blobs and apply a linear transform 

Indsutry use: biological taxonomies (cell → tissue → organ); customer segments with sub‑segments.

4. Nested (hierarchical) clusters
    - A tight gaussian inside a looser gaussian, inside a still looser one

Industry use: biological taxonomies (cell → tissue → organ); customer segments with sub‑segments.

5. High dimensional clusters on manifolds 
    - Swiss‑roll or S‑curve in 3D, projected into 10+ dims with noise

Industry use: user‐behavior time‑series that lie on low‑dimensional manifolds; word‐embedding clusters in NLP.

6. Heavy‑tailed / outlier‐rich clusters
    - Gaussian core plus a few Cauchy or t‑distributed outliers

Industry use: financial transactions with occasional extreme events; network‐traffic bursts.
