## Code konzept

### Es gibt als erstes einen parameter Eh, der so eine art gamma von MMDEW ist. also man nimmt die pre change daten und de "kernel bandwith" dafür.
In summary, the function Eh_square seems to be a measure of similarity or dissimilarity between different partitions of the pre_change_sample data, after applying the Gaussian RBF kernel transformation.

### Danach berechnet man die Covh, auch eine aussage über das prechange sample.
This might be used to measure how similar or dissimilar the changes between different partitions are in the transformed (kernel) space.

### Dann macht man einen vector lauter 0en der länge der verschiedenen block größen namens variance_est

### Dann geht man durch alle verschiedenen Blockgrößen durch und und berechnet:
Variance Estimate Calculation:
Eh_sq./Num_blk: This is dividing the value of Eh_sq by Num_blk. This could be seen as a way of normalizing or averaging the measure over the number of blocks.
1-1./Num_blk: This operation subtracts the reciprocal of Num_blk from 1. If Num_blk is large, this value will be close to 1. If Num_blk is small, it will be farther from 1. This acts as a weight.
.*Covh: This multiplies the weighted value from the previous step with Covh. So, the contribution of Covh to the final value is modulated by this weight.
(Eh_sq./Num_blk + (1-1./Num_blk).*Covh): This is the sum of the normalized Eh_sq and the weighted Covh.
nchoosek(omega_B(i),2): This computes the binomial coefficient, commonly known as "n choose k". It's the number of ways to choose 2 items from omega_B(i) items without repetition and without order. This can be seen as a scaling or normalization factor depending on the current value of omega_B(i).
./nchoosek(omega_B(i),2): The whole previously computed sum is then divided by this binomial coefficient. N choose k => omega_B(i) = N
Overall, this loop is calculating a measure of variance (or some other form of spread or dispersion) for each block size in omega_B. This measure is based on the interplay between Eh_sq and Covh, and is normalized by factors related to the number of blocks and the current block size from omega_B.

The result, variance_est, will be an array where each element is this computed measure for a corresponding block size in omega_B.
also berechnen wir hier die variance für alle verschiedenen blockgrößen ig


