## To do
- Set up a virtual environment for this specific project - ✅
- Make datasets actually noisy, and do tests with increased noise and blobs - ✅ 
- Look at notes you took during meeting with Professor Smith - ✅ 
- Do step 0 that professor smith mentioned - ✅ 
- Try out step 0 with some tests - 
- Read and understand documents last sent by Prof smith - 
- Implement deconvolution - 
- Maybe work with step 1 and 2 - 
- What exactly are mixture models? - 
- Start trying the abstract approaches that were pitched or new ones - 

### something we have not yet explored is changing the type of noise added to the blobs, as of right now we have only used gaussian noise

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