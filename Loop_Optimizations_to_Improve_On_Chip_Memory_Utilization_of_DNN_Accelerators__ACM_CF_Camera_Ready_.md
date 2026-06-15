## **DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators** 

Piyumal Ranawaka Muhammad Waqar Azhar Per Stenstrom piyumal@chalmers.se waqarm@chalmers.se per.stenstrom@chalmers.se Chalmers University of Technology Chalmers University of Technology Chalmers University of Technology Gothenburg, Sweden Gothenburg, Sweden Gothenburg, Sweden 

## **ABSTRACT** 

Deep neural network (DNN) accelerators suffer from poor utilization of on-chip memory which potentially reduces performance and energy efficiency. Loop reordering and blocking are used to improve on-chip memory utilization in DNN accelerators. However, existing optimization frameworks are inefficient due to either a prohibitive time complexity of searching the entire search space or due to a sub-optimal choice of optimizations. 

This paper proposes DNNOPT – a hardware/software framework for optimally selecting loop order and blocking factors, for loop reordering and blocking in isolation or in combination. DNNOPT uses proposed _Early exit_ and _Strided search_ strategies to prune the search space and simple analytical models of data reuse to evaluate each optimization point efficiently and accurately. Overall, DNNOPT reduces the search space by more than two orders of magnitude and improves performance, energy efficiency and time to solution, on average, by 1 _._ 8×, 50%, and 226×, respectively, of convolutional neural network (CNN) and Transformer applications compared to state-of-the-art frameworks. 

## **CCS CONCEPTS** 

• **Computer systems organization** → **Architectures** ; _Embedded and cyber-physical systems_ . 

## **KEYWORDS** 

DNN acceleration, Loop Re-Order, Loop Blocking, Reuse Distance, Energy Efficient DNN Acceleration, On-chip Memory Management 

## **ACM Reference Format:** 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom. 2024. DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators. In _21st ACM International Conference on Computing Frontiers (CF ’24), May 7–9, 2024, Ischia, Italy._ ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3649153.3649196 

## **1 INTRODUCTION** 

Many applications need Deep Neural Networks (DNN) today and are compute and data intensive making them a target for acceleration. However, the memory system in DNN accelerators is typically 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _CF ’24, May 7–9, 2024, Ischia, Italy_ 

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0597-7/24/05...$15.00 https://doi.org/10.1145/3649153.3649196 

a performance and energy bottleneck, where off-chip memory accesses dominate. The key to decreasing memory accesses is to maximize the reuse of data in the on-chip memory. 

_Loop reordering_ and _blocking_ are well-studied optimizations considered to maximize data reuse in DNN accelerators. However, prior art has failed to come up with efficient frameworks to select loop order and blocking factors for DNN accelerators. Specifically, frameworks in prior art suffer from either (1) having to search a prohibitively large search space exhaustively, (2) not guaranteeing an optimal choice of optimizations or (3) computationally demanding evaluation of each point, be it loop order or blocking factors. 

First, [34, 35, 39, 41, 43, 53, 57, 60, 61] _exhaustively_ find the optimization that minimizes the execution time over a prohibitively large search space with _𝑂_ ( _𝑀_ × _𝑃_ ) computational complexity [16]. Here, _𝑀_ is the number of iterations to evaluate a single point in the search space, and _𝑃_ is the number of points in the search space. Some strides have been taken to reduce the search space [16, 20, 26, 35, 51]. However, it is still prohibitively large given that all possible sizes of blocking factors and their combinations should be considered. Other work has deployed feedback-driven approaches with black-box auto-tuning, beam search, Monte Carlo, or machine-learning-based algorithms using iterative sampling [2, 17, 27, 42, 61]. However, they suffer from a high training cost or necessitate massive datasets and large-scale simulations to train performance models [26]. 

Second, Timeloop [24, 39] uses random search and [28, 63] use genetic algorithms. Unfortunately, these methods lead to _non-optimal_ results. For example, Timeloop’s sub-optimal random search method may choose points in the search space that yield five times worse performance than the optimal point in the search space [26]. 

Third, to evaluate a single point in the search space, Timeloop [39] visits all the elements in two contiguous tiles for data reuse. Rahaman et al. [43] analyze each location of the on-chip memory with a simulation. This leads to a high computational cost in evaluating the cost of each individual point in the search space. 

This paper proposes DNNOPT – a hardware/software framework to choose loop order and blocking factors, optimally and efficiently. The first challenge is how to reduce the search space without compromising global optimality. A first strategy we employ is _Early exit_ from the search process, if the number of off-chip accesses does not decrease with respect to increasing a blocking factor. A second strategy is _Strided search_ , where the idea is to stride through the search space visiting only valid mappings given the systolic array-based architecture. The second challenge is reducing the computational complexity of evaluating a single point in the search space. The proposed method uses precise and computationally inexpensive analytical cost models based on reuse distance and inter-block reuse to capture the locality inspired by [40, 50]. 

CF ’24, May 7–9, 2024, Ischia, Italy 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

DNNOPT is used to quantitatively compare the merits of the selected optimizations, for loop reordering and blocking in isolation or in combination. DNNOPT also selects a loop order and blocking factors layer-by-layer. Our contributions are as follows: 

- We propose DNNOPT – a hardware/software framework for systolic array-based accelerators that selects the loop order and blocking factor, layer-by-layer, in isolation or in combination. 

- We show that DNNOPT selects an optimization that can optimally minimize the number of off-chip memory accesses with substantially lower computational complexity using two proposed techniques – _Early exit_ and _Strided search_ and with the use of simple and computationally inexpensive analytical cost models for systolic array based accelerators. 

- We evaluate the merits of the framework. We show that we can prune the search space by more than 99%, i.e., two orders of magnitude, for CNN and transformer workloads with the proposed methods. Further, our framework leads to 1.8×, 50%, and 249× improvement in performance, energy efficiency and time to solution respectively, on average, compared to Timeloop [39] random search, and 226× estimated improvement in time to solution compared to CoSA [26] for CNN and Transformer benchmarks. 

As for the rest of the paper, we introduce key concepts in Section 2. Then, Section 3 presents our proposed techniques. We move to the experimental results by introducing our methodology in Section 4 followed by the results in Section 5. Finally, we put our work in the context of others in Section 6 before we conclude in Section 7. 

## **2 BACKGROUND** 

Section 2.1 introduces the convolution algorithm comprising the use case. Then, loop reordering and blocking and the baseline architecture are introduced in Sections 2.2 and 2.3, respectively. 

## **2.1 Convolution and Loop Nest Representation** 

The convolution algorithm can be represented as six nested loops in output-centric form as shown in Algorithm 1. To compute a single element in the output feature map (OFMAP), the dot product between a weights filter and a portion of the input feature map (IFMAP) with identical dimensions to the weights is performed as shown in Figure 1 (top). Next, the weights filter is swept across the entire IFMAP. The IFMAP has a width of _𝑊_ , a height of _𝐻_ , and a depth of _𝐶_ . Similarly, a weights filter has a width of _𝑅_ , a height of _𝑆_ , and a depth of _𝐶_ , and there are _𝑁_ weights filters resulting in _𝑁_ channels along the depth for the OFMAP. 

The dimensions of a layer and compute tile for output stationary systolic arrays are shown in Figure 1. Computation is split into compute tiles that fit a systolic array of size _𝐴_ × _𝐴_ after the _im2col_ transformation. The stride of a convolution is denoted by _𝑇_ . 

The three innermost loops in Algorithm 1 (lines 4 – 6) are hardwired for a systolic array with output stationary dataflow. This corresponds to moving across the width, height, and depth of weights. The three outermost loops (lines 1 – 3) correspond to moving along the OFMAP’s width, height, and depth. 

**Figure 1: The layer topology and compute tiles.** 

We will refer to the order of the three outermost loops as _loop order_ . For example, the XYZ loop order refers to traversing along the width, height, and depth of the OFMAP, in that order, whereas the XZY loop order refers to traversing along the width, depth, and height of the OFMAP. 

**Algorithm 1** Convolution algorithm for loop order XYZ. 

- 1: **for** Z=0,Z<N,Z++: **do** 2: **for** Y=0,Y<H,Y++: **do** 3: **for** X=0,X<W,X++: **do** 4: **for** c=0,c<C,c++: **do** 5: **for** r=0,r<R,r++: **do** 6: **for** s=0,s<S,s++: **do** 7: O[Z][Y][X]+= 8: W[Z][c][r][s]*I[c][Y+r][X+s] 9: **end for** 

- 10: **end for** 11: **end for** 12: **end for** 13: **end for** 14: **end for** 

## **2.2 Loop Reordering and Blocking** 

Loop reordering (or loop interchange) changes the order of the three outermost loops shown in Algorithm 1. It impacts the temporal locality of data access and, consequently, the data reuse distance. 

Loop blocking refers to partitioning the data accessed by loops into blocks that fit the size of the on-chip memory hierarchy. The three outermost loops in Algorithm 1 plow through the OFMAP. The OFMAP is partitioned into blocks of size _𝐵ℎ_ × _𝐵𝑤_ × _𝐵𝑛_ , where _𝐵ℎ_ , _𝐵𝑤_ , and _𝐵𝑛_ represent the height, width, and depth, respectively, of each block. 

DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators 

CF ’24, May 7–9, 2024, Ischia, Italy 

be obtained from off-chip memory and the TLB as well as the MAT will be updated and the previous address mapping is invalidated. The combination of the page table and the MAT makes the on-chip memory fully associative. 

## **3 DNNOPT: PROPOSED FRAMEWORK** 

**Figure 2: The baseline architecture.** 

## **2.3 Baseline Architecture** 

Figure 2, depicts the assumed baseline architecture with blue-shaded boxes. Inspired by the Pascal architecture [12], it uses an output stationary systolic array as the compute core. It has unified on-chip memory for inputs and weights. In an output stationary systolic dataflow, the input and weight fetchers (IF and WF) stream in the input and weight tiles from left and from the top, respectively(see bottom part of Figure 1). A partial sum is accumulated, temporally, on the Processing Elements (PE). When accumulation completes, the output writer (OW) writes the output tile to memory. 

The baseline accelerator is connected to the host as a slave device. The DNN compiler runs on the host and compiles a given DNN prior to inference. Input to the compiler is the DNN topology (the dimensions of IFMAP, OFMAP, the weights, and stride, i.e., H, W, C, R, S, N, and T in Section 2.1), and microarchitectural parameters such as the size of the PE array and the on-chip memory. 

The memory hierarchy of the accelerator comprises two levels: on-chip and off-chip memory. The address space is subdivided into fixed-size _pages_ . A page table located in the off-chip memory keeps track of where in the on-chip memory a page is allocated, if at all. A Translation Lookaside Buffer (TLB), is used to cache the most recently used page translations as shown in Figure 2. Pages are pre-loaded from off-chip memory to on-chip accelerator memory. To keep track of free space, the on-chip memory is managed as two circular buffers for weights and inputs where a pointer refers to the next page frame to load data. When a page is inserted, the page table will be updated with the location of that page. In the baseline architecture, the unified on-chip memory is divided logically into two fixed parts across all layers for weights and inputs. Each of them has a circular pointer pointing to where to load the next page in the on-chip memory. 

On a first access to a page, the page will be pre-loaded into a location in the on-chip memory dictated by the circular pointer and the on-chip location will be cached in the TLB. Subsequent accesses to said page can be satisfied by a lookup in the TLB. Since the on-chip memory is managed as a circular buffer, pages will get overwritten when the on-chip memory is filled up. As a consequence, more than one page can map into the same on-chip slot. To keep track of which page currently resides in a page slot in the on-chip memory, the baseline architecture has a Memory Allocation Table (see Figure 2). The MAT has one entry per on-chip memory page slot storing the off-chip page address as a tag. On a tag mismatch, the page must 

This section first provides a detailed exposition of the design of the software support in Sections 3.1–3.3 and then the required hardware architecture support in Section 3.4. The DNNOPT framework is shown in Figure 3. The software support is shown on the left and the hardware support on the right. The components which are unmodified, modified, and added, with respect to the baseline, are marked in blue, orange, and gray, respectively. 

## **3.1 Loop Reordering** 

We show how loop reordering in isolation can be deployed with constant-time computational complexity and achieve optimal results. For this, we use Global Reuse Distance (GRD) to establish the amount of on-chip memory required to capture full reuse. GRD with respect to a data structure D is defined as the number of unique accesses between the first and last access to D. We establish GRD, for each layer, for all six loop orders, for weights and inputs to determine the loop order that maximizes GRD per layer. The minimum on-chip memory required to capture full reuse is determined by the sum of GRD for weights and inputs. We ignore the size of the output memory as it is fixed in output stationary dataflow. 

If the minimum total memory for a given loop order is less than or equal to the total available on-chip memory, the layer fits in the on-chip memory upon reordering. In this case off-chip memory accesses are limited to the compulsory off-chip accesses (theoretical minimum/optimum). Furthermore, deriving the optimum loop order has a constant-time computational complexity ( _𝑂_ (1)) as compared to _𝑂_ ( _𝑀_ × _𝑃_ ) for exhaustive search. Here, _𝑀_ is the computational complexity of evaluating a given point in the search space and _𝑃_ is the number of points in the search space. 

GRD depends on the loop order. To see this, let us consider two example loop orders, XYZ and XZY, out of the six possible looporder permutations. Figure 4(a) shows GRD for loop order XYZ. The IFMAP is traversed in the X and the Y loops with a set of weights, prior to traversing across all weights in the Z loop. Therefore, the IFMAP and a set of weights, dictated by the size of the systolic array, must be accommodated. Hence, GRD for inputs is H × W × C corresponding to the entire IFMAP and for weights it is R × S × C × A (see Section 2.1). 

For loop order XZY, as shown in Figure 4(b), the IFMAP is traversed along the width with the X loop and across all the weights with the _𝑍_ loop and finally along the height in the Y loop. Hence, the entire set of weights of size _𝑅_ × _𝑆_ × _𝐶_ × _𝑁_ and a portion of the IFMAP of size _𝑆_ × _𝑊_ × _𝐶_ , must be accommodated. Table 1 shows the derived analytical models for GRD for weights (W) and inputs (I) for all six loop orders. The same methodology is applied to derive GRD for XYZ and XZY. 

## **3.2 Loop Blocking** 

Prior art determines the blocking factor for input and weights based on the size of the on-chip memory. Figure 5 shows where the 

CF ’24, May 7–9, 2024, Ischia, Italy 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

**Figure 3: The baseline architecture corresponds to the blue-shaded boxes. DNNOPT extends the baseline for layer-by-layer loop reordering, blocking, and memory allocation with modified boxes (orange) and new functionalities (gray boxes).** 

with the IFMAP block dimensions, we ignore the convolutional overlap between neighboring blocks. 

**Figure 5: Blocking of a convolution layer.** 

**Figure 4: (a) GRD for the XYZ loop order (b) GRD for the XZY loop order.** 

**Table 1: Global reuse distance (GRD) for weights and inputs.** 

|**Loop Order**|**GRD (W)**|**GRD (I)**|
|---|---|---|
|XYZ<br>XZY<br>YZX<br>YXZ<br>ZYX<br>ZXY|R×S×C×A<br>R×S×C×N<br>R×S×C×N<br>R×S×C×A<br>R×S×C×N<br>R×S×C×N|H×W×C<br>S×W×C<br>H×C×A<br>H×W×C<br>H×C×A<br>R×W×C|



selected block size for the IFMAP is _𝐵ℎ_ × _𝐵𝑤_ × _𝐶_ , and the block size for weights is _𝑅_ × _𝑆_ × _𝐶_ × _𝐵𝑛_ . IFMAP is divided into four blocks I1–I4, weights into two, i.e. W1 and W2, and output into eight, i.e., O1–O8. As filter dimensions are often negligible in comparison 

With the aforementioned blocking approach, inspired by Peemen et al. [40], the number of memory accesses is given by the product of the block size and the number of block accesses. Under the XYZ loop order, the access sequence for the example in Figure 5 is I1, W1, I2, W1, I3, W1, I4, W1, I1, W2, I2, W2, I3, W2, I4, W2. We note that each weight block is reused four times by exploiting interblock temporal locality. Hence, the number of memory accesses for weight blocks ( _𝑀𝐴𝑊_ ) is 

**==> picture [210 x 21] intentionally omitted <==**

Here, we assume that _𝑁_ is an integer multiple of _𝐵𝑛_ and that the number of memory accesses for IFMAP blocks (MA _𝐼_ ) is _𝑊_ and _𝐻_ and are integer multiples of _𝐵𝑤_ and _𝐵ℎ_ , respectively. 

**==> picture [242 x 30] intentionally omitted <==**

The question is whether one can reduce the number of memory accesses by changing the block parameters: _𝐵ℎ, 𝐵𝑤_ , and _𝐵𝑛_ . Considering weight blocks first, we note that increasing/decreasing _𝐵𝑛_ has no effect on the number of memory accesses, according to Equation 1. Similarly, for IFMAP blocks, the number of memory accesses is affected only by _𝐵𝑛_ , according to Equation 2. By increasing _𝐵𝑛_ , 

DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators 

CF ’24, May 7–9, 2024, Ischia, Italy 

|**Algorithm 2**Blockingspace search.|**Algorithm 2**Blockingspace search.|
|---|---|
|1:|**for**layer∈DNN**do**|
|2:|ofchip_accessesmin|
|3:|←ESTIMATE_OFFCHIP_ACCESSES(A,1,A)|
|4:|**for**Bn=A,Bn<N,Bn+A**do**|
|5:|**for**Bh=1,Bh<H,Bh++**do**|
|6:|**for**Bw=A,Bw<W,Bw+A**do**|
|7:|ofchip_accesses|
|8:|←ESTIMATE_OFFCHIP_ACCESSES(Bw,Bh,Bn)|
|9:|size_𝑖_←C*Bh*Bw|
|10:|size_𝑤_←C*R*S*Bn|
|11:|if(size_𝑖_+size_𝑤<_ (Memory_𝑖_+Memory_𝑤_)/2)|
|12:|if(ofchip_accesses_<_ofchip_accessesmin)|
|13:|Opt_Bn←Bn|
|14:|Opt_Bh←Bh|
|15:|Opt_Bw←Bw|
|16:|ofchip_accessesmin ←ofchip_accesses|
|17:|**end for**|
|18:|**end for**|
|19:|**end for**|
|20:|**end for**|



we can reduce the number of memory accesses. At first glance, this is not possible as _𝐵𝑛_ is bounded by the size of the on-chip memory. However, an important and interesting insight is that by compensating a larger _𝐵𝑛_ for a smaller _𝐵ℎ_ and _𝐵𝑤_ , we can find other sets of blocking parameters that minimize the number of memory accesses without being bounded by the size of the on-chip memory. This analytical approach of on-chip memory locality analysis is used to lower the computational complexity of evaluating a given point in the search space. Evaluating a single point in the search space is done in constant time and has a computational complexity of O(1). Next, our approach to establish the set of blocking parameters that minimizes the number of memory accesses, for each layer, is to design an algorithm that evaluates all parameter settings and their combinations. This is specified in Algorithm 2. 

**Strided search** : For each layer (line 1 in Algorithm 2), valid settings of _𝐵ℎ_ , _𝐵𝑤_ , and _𝐵𝑛_ are evaluated by iteratively sweeping through all valid block sizes and their combinations (lines 4 – 6). Given an output stationary systolic array architecture that computes the DNN, tile by tile, block sizes in the X and the Z dimensions which are not multiples of A (the size of the PE array), are not valid. Therefore, instead of traversing such loops with unit steps, striding with a stride of A will reduce the computational complexity of the search. Here, we propose _strided search_ for two of the loops (lines 4 and 6 ), which are initialized and incremented with A (the PE array size) to respect the output stationary systolic array’s compute tiles. 

In Algorithm 2, offchip_accessesmin keeps the minimum number of off-chip accesses achieved and is initialized to that of the first blocking candidate (lines 2 and 3). For each blocking candidate, the number of memory accesses is established using the cost models according to Equations 1 and 2 (lines 7 and 8). Next, the size of the input and weight blocks (lines 9 and 10) is calculated. If the total size of the blocks is less than the available on-chip memory, it is a valid candidate (line 11). The available on-chip memory is dictated by the size of the input (Memory _𝐼_ ) and the weights (Memory _𝑊_ ). 

We assume that double buffering is used so that the size is half of the available amount of on-chip memory (line 11). Now, if the total number of off-chip accesses achieved by this candidate is less than the minimum number of off-chip accesses in the search so far, the current candidate is selected as the best candidate, and this updates the records (lines 12 – 16). 

## **3.3 Combined Loop Reordering and Blocking** 

We now combine layer-by-layer loop reordering with blocking. In the previous section, we show how to establish the number of memory accesses for a certain block size given an XYZ loop order. We have used the same methodology to derive analytical cost models for the number of memory accesses for the other five loop orders. Table 2 lists the analytical models for off-chip accesses for inputs (I) and weights (W) for all six inter-block loop orders. The ceiling function establishes the exact number of memory accesses. 

We can trivially extend Algorithm 2 to establish the loop order and the blocking factor, per layer, that minimizes the number of memory accesses. This is done by using a relevant cost model according to Table 2 in the algorithm to estimate the number of off-chip accesses (lines 7 and 8). 

The combined loop reordering and blocking technique use early exit to prune the search space. To explain how, note that the analytical cost models shown in Table 2 can be further simplified by ignoring the ceiling function similar to Equations 1 and 2. The ceiling function only serves to account for the last blocks at the edge when the layer dimension is not a multiple of the block size. **Early exit** : Based on the simplified cost models we note that the numbers of off-chip accesses are monotonic functions with respect to a given blocking factor. It is true as long as the number of off-chip accesses is dependent of the size of blocking factor. To understand this let’s take the same example as in Equations 1 and 2. The number of off-chip accesses for weights is independent of blocking factor sizes whereas the number of off-chip accesses for inputs monotonically decreases when the size of a blocking factor keeps increasing. Therefore, if the number of off-chip accesses does not improve when increasing a blocking factor, the increased blocking factor has no impact on off-chip accesses and considering it is therefore unproductive. Similarly, when _𝐵𝑛_ increases, the number of off-chip accesses will decrease monotonically. However, when _𝐵ℎ_ or _𝐵𝑤_ increase, there will be no improvement of off-chip accesses between consecutive iterations. Therefore, search along that path could be terminated early. As we will show experimentally, the Early exit strategy can significantly reduce the search time. 

The implementation of Early exit is shown in Algorithm 3. It is built on top of Algorithm 2. Code extensions to Algorithm 2 are highlighted in colors (marked with green for initializing new variables, red for terminating loops and blue for updating). To determine if the number of off-chip accesses of the current iteration of a given loop is less than that of the previous iteration, we need to save the number of off-chip accesses of the previous iteration for a given loop. For this, we use three variables prev_offchip_n, prev_offchip_h and prev_offchip_w for the loops _𝑁_ , _𝐻_ and _𝑊_ , respectively. They are initiated to infinity before the invocation of each loop (lines 4, 6 and 8 and denoted in green). Further, they are updated at the end of each iteration of each loop (lines 22, 26, and 

CF ’24, May 7–9, 2024, Ischia, Italy 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

## **Algorithm 3** Early Exit Search. 

- 1: **for** layer ∈ DNN **do** 2: offchip_Accessesmin 3: ← ESTIMATE_OFFCHIP_ACCESSES(A,1,A) 4: prev_offchip_n ←∞ 5: **for** Bn=A,Bn<N,Bn+A **do** 6: prev_offchip_h ←∞ 7: **for** Bh=1,Bh<H,Bh++ **do** 8: prev_offchip_w ←∞ 9: **for** Bw=A,Bw<W,Bw+A **do** 

- 10: offchip_accesses ← 11: ESTIMATE_OFFCHIP_ACCESSES(Bw,Bh,Bn) 12: size _𝑖_ ← C ∗ Bh ∗ Bw 13: size _𝑤_ ← C ∗ R ∗ S ∗ Bn 14: if (size _𝑖_ + size _𝑤 <_ (Memory _𝑖_ + Memory _𝑤_ )/2) 15: if(offchip_accesses _<_ offchip_Accessesmin) 16: Opt_Bn ← Bn 17: Opt_Bh ← Bh 18: Opt_Bw ← Bw 19: offchip_accessesmin ← offchip_accesses 20: if (prev_offchip_w _<_ = offchip_accesses) 21: break 22: prev_offchip_w ← offchip_accesses 23: **end for** 24: if (prev_offchip_h _<_ = offchip_accesses) 25: break 26: prev_offchip_h ← offchip_accesses 27: **end for** 28: if (prev_offchip_n _<_ = offchip_accesses) 29: break 30: prev_offchip_n ← offchip_accesses 31: **end for** 32: **end for** 

**Table 2: Analytical cost models for off-chip accesses for blocking combined with loop reordering.** 

|**Inter Block**<br>**Loop Order**|**of-chip**<br>**accesses (I)**|**of-chip**<br>**accesses (W)**|
|---|---|---|
|XYZ<br>YXZ|_𝐵𝑤_×_𝐵ℎ_×_𝐶_×<br>⌈_𝐻_<br>_𝐵ℎ_⌉⌈_𝑊_<br>_𝐵𝑤_⌉⌈_𝑁_<br>_𝐵𝑛_⌉|_𝑅_×_𝑆_×_𝐶_×_𝐵𝑛_×<br>⌈_𝑁_<br>_𝐵𝑛_⌉|
|ZXY<br>ZYX|_𝐵𝑤_×_𝐵ℎ_×_𝐶_×<br>⌈_𝐻_<br>_𝐵ℎ_⌉⌈_𝑊_<br>_𝐵𝑤_⌉|_𝑅_×_𝑆_×_𝐶_×_𝐵𝑛_×<br>⌈_𝐻_<br>_𝐵ℎ_⌉⌈_𝑊_<br>_𝐵𝑤_⌉⌈_𝑁_<br>_𝐵𝑛_⌉|
|XZY<br>YZX|_𝐵𝑤_×_𝐵ℎ_×_𝐶_×<br>⌈_𝐻_<br>_𝐵ℎ_⌉⌈_𝑊_<br>_𝐵𝑤_⌉⌈_𝑁_<br>_𝐵𝑛_⌉|_𝑅_×_𝑆_×_𝐶_×_𝐵𝑛_×<br>⌈_𝑊_<br>_𝐵𝑤_⌉⌈_𝑁_<br>_𝐵𝑛_⌉|



## **4 EXPERIMENTAL METHODOLOGY** 

## **4.1 Simulation Setup** 

We extend the SCALE-Sim [47] cycle-accurate simulator to model the proposed optimizations. The baseline system (Section 2.3) uses fixed memory allocations of on-chip memory of size 256 KB for input and 128 KB for weights, for a total of 384 KB, and a 32x32 output stationary PE array. The size of the off-chip memory is 64 MB to suit all DNNs under consideration. It is divided into pages of size 1 KB. A TLB of size 1 KB is used to cache the most recent page translations. The MDT is 2 KB. 

As in Pascal [12], we assume a 22-nm technology node and a clock frequency of 1 GHz. A more aggressive technology node would improve the results since it would reduce the on-chip energy costs and latencies. As we target power-constrained devices, LPDDR4 is assumed as the off-chip technology across all experiments with a bandwidth of 32 GB/s. 

## **4.2 Benchmarks** 

30 are denoted in blue). Finally, if the number of off-chip accesses for each loop has not improved with respect to the last iteration of the same loop we break the loop (lines 20, 21, 24, 25, 28, and 29, denoted in red), terminating the search early. 

## **3.4 Architecture and Hardware Support** 

The baseline architecture, according to Section 2.3, is extended with a _metadata table_ (MDT) with one row per layer (see Figure 3). The metadata with the selected per layer optimization and the base addresses for the input, weight and output compute tiles (shown with dots in Figure 1) is established at compile time and transferred to the MDT prior to execution. The control logic of the accelerator has a pointer to the MDT. 

Reading the MDT and base address metadata introduces a latency. However, this can be effectively hidden by overlapping it with the computation of the last compute tile of the previous layer. Further, to change the layer-by-layer optimization while maintaining high utilization, a fully associative memory is needed. This is provisioned by the proposed baseline on-chip memory management architecture (see TLB and MAT in Section 2.3). 

We choose a wide variety of DNN benchmarks shown in Table 3 selected from MLPerf [44], Deep Bench benchmarks [1], and from benchmarks used in previous work [18, 19]. 

## **4.3 Metrics** 

For performance, we use _speedup_ , the ratio of execution time of the baseline system to the proposed systems. We use _percentage energy reduction_ to quantify energy-efficiency improvements. The dynamic energy cost and static energy cost per cycle of each component is obtained from CACTI [28] and Accelergy [59]. These costs are shown in Table 4. Then We derive dynamic energy by multiplying the memory-access counts to each level in the memory hierarchy and the number of MAC operations performed, with the dynamic energy costs shown in Table 4. We derive static energy by multiplying the static energy cost per cycle of each component, shown in Table 4 with the total number of cycles. By summing up all components, we can establish the total energy. 

We use _Improvement factor of time to solution_ as a measure of search efficiency. Both the proposed framework and the baseline are evaluated on the same computer. It has an Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz with 2 cores with a 4-MB cache. 

DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators 

CF ’24, May 7–9, 2024, Ischia, Italy 

**Table 3: Benchmark DNNs. References are provided to the original papers of the benchmarks.** 

temporal locality and blocking temporal as well as spatial locality the combination yields the highest improvement due to synergies. 

|**Category**<br>~~|~~|**Workload**<br>~~|~~|
|---|---|
|Image Classification & Recognition<br>~~|~~|ResNet50 [58]<br>GoogLeNet [23]<br>MobileNet v1 [25]<br>Deepbench-Face Recognition[1]<br>~~|~~|
|Object Detection|YOLO-Tiny [15]|
|Semantic Segmentation|FasterRCNN[45]|
|Speech Recognition|Deep-speech[1]|
|Language and Translation|Deep Bench LSTM [1]<br>Deep Bench-Machine Translation [1]<br>Transformers[56]|



**Table 4: Energy costs.** 

|~~a~~|~~a~~|~~|~~|
|---|---|---|
|**Operation**<br>~~a~~|**Energy Cost of Access(pJ)**<br>~~a~~|**Static Energy per Cycle(pJ)**<br>~~|~~|
|MAC Operation<br>On-Chip Memory Access<br>LPDDR Off-chipAccess<br>~~a~~<br>~~ee~~|0.185<br>12.411<br>512<br>~~a~~<br>~~ee~~|0.002<br>0.052<br>25324.8<br>~~|~~|



**Figure 6: Percentage off-chip access reduction.** 

## **4.4 Modeled Systems** 

We use two baselines to compare against: 

- **Baseline System 1 (BS1)** . BS1 is modeled according to Section 2.3 with no loop optimizations. BS1 serves as a baseline to systematically study the impact of each loop optimization. 

- **Baseline System 2 (BS2)** . BS2 uses BS1 with loop reordering and blocking candidates per layer selected by Timeloop Mapping (BS2) [39] with ruby imperfect factorization [24]. We use Timeloop’s Random search, which is non-exhaustive and leads to a reasonable time complexity. Search is terminated if 500 consecutive sup-optimal mappings or 1500 consecutive invalid mappings are found which is similar to the search criteria used in the CoSA[26] evaluation. For few of the layers, which do not produce any valid mappings within the above search criteria, we use the baseline mapping of no loop reordering or blocking. 

LSTM and machine translation see no improvement because their footprints fit the size of on-chip memory. Deep speech suffers from 15% more accesses for loop reordering and blocking and 5% for the combined optimization. The reason is that accesses to memory is done at page granularity. If at least a single data element in a page is needed the entire page needs to be pre-loaded. 

## **5.2 Impact on Performance** 

Figure 7 shows the speedup obtained for each of the optimizations normalized to the baseline (BS1). Speedup, in general, follows the trends of the reduction of memory accesses in Figure 6 and all, except for the three rightmost benchmarks, benefit from reduced memory accesses. Loop reordering provides a speedup of 1 _._ 1×–1 _._ 8×, blocking offers a higher speedup (1 _._ 3×–2×) whereas the combination yields a higher speedup of 1 _._ 4× to 3× due to synergy. As for the three rightmost benchmarks, we see, as expected, no speedup as the number of memory accesses is not reduced. 

## **5.3 Impact on Energy Efficiency** 

## **5 EVALUATION** 

This section presents our results focusing on the relative performance of loop reordering, blocking and their combination in Sections 5.1 to 5.3 and comparison with state-of-the-art in Sections 5.4 and 5.5. Finally, we perfrom a sensitivity analysis in Section 5.6. 

## **5.1 Reduction of Memory Accesses** 

Figure 6 shows the reduction of memory accesses (y-axis) normalized to the baseline (BS1) for the three proposed optimizations for all benchmarks (x-axis). Except for LSTM, machine translation and deep speech, all optimizations help reducing the number of memory accesses by between about 10% to 80% where loop reordering improves by 10%–60%, blocking improves by 20%–70% and the combination improves by 30%–80%. As loop reordering leverages 

Figure 8 shows the reduction in energy (in percent) for the three optimizations relative to the baseline (BS1) across the benchmarks. Overall, the relative improvements are consistent with the reduction in memory access according to Figure 6 as expected as off-chip accesses are costly. Specifically, loop reordering cuts energy by 5% to 50%, blocking by 20% to 55% and the combination by 25% to 70% with an average of 50%. The three rightmost applications, again, show no improvement as expected. On the contrary, deep speech shows some slight deterioration of energy efficiency following the trends in memory accesses as noticed earlier. 

## **5.4 Reduction of Search Space** 

Table 5 shows the average percentage reduction of the search space achieved by **Strided search** and **Early exit search** in isolation 

CF ’24, May 7–9, 2024, Ischia, Italy 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

**Table 5: Average percentage search space reduction across layers.** 

|**Workload**<br>**DNN**<br>~~ee~~|**Strided**<br>**Search**<br>~~eee~~|**Early Exit**<br>**Search**<br>~~eee~~|**Combined**<br>~~eee~~|
|---|---|---|---|
|DB Face Recognition<br>Faster RCNN<br>GoogLeNet<br>MobileNet<br>ResNet<br>YOLOTiny<br>Transformer<br>LSTM<br>Machine Translation<br>~~ee~~|99.7%<br>99.8%<br>99.7%<br>96.9%<br>99.7%<br>99.7%<br>99.7%<br>30.0%<br>30.0%<br>~~eee~~|96.7%<br>99.9%<br>90.8%<br>91.6%<br>97.8%<br>97.0%<br>97.64%<br>0.0%<br>0.0%<br>~~eee~~|99.9%<br>99.9%<br>99.9%<br>99.5%<br>99.9%<br>99.7%<br>99.0%<br>30.0%<br>30.0%<br>~~eee~~|



**Figure 7: Speedup obtained by proposed techniques.** 

**Figure 9: Performance and percentage energy reduction for the proposed optimizations over state of the art.** 

**Figure 8: Energy reduction for the proposed optimizations.** 

and in combination. Both techniques can prune the search space effectively yielding almost two orders of magnitude reduction of the search space for most of the applications. On average, the search space reduction for **Strided search** is 99.3%, for Early exit it is 95.2% and it is more than 99% for the combination across all layers of all the CNN workloads. This means that we prune the search space by two orders of magnitude Unfortunately, **Early exit search** is ineffective for LSTM and Machine translation and marginal improvements are obtained using **Strided search** (30%). 

## **5.5 Comparison with TimeLoop and CoSA** 

Previously, a given technique among loop reordering in isolation, blocking in isolation, or a combination of the two were used across all layers. However, now we pick the best technique among three optimizations for each layer and compare it with the mapping chosen by TimeLoop [39]. 

Figure 9 shows the speedup (left y axis) and reduction in energy (right y axis) for DNNOPT when it picks the best optimization per layer. The results are normalized to BS2 that selects the best mapping per layer using Timeloop [39]. Speedup ranges from 1 _._ 2× to 2 _._ 4× for the CNN and transformer applications and energy is reduced by between 17% and 61%. The speedup and percentage energy reduction, on average, is 1.8x and 50%, respectively, as compared to Timeloop for CNN and Transformer workloads. The two rightmost applications show no improvement as expected due to small layers that readily fit in on-chip memory. The speedup and energy reductions are due to the fact that Timeloop does not make optimal choices where as DNNOPT is optimal. Further, DNNOPT picks the optimal optimization per layer without doing exhaustive search. 

Figure 10 shows the improvement factor of time to solution when using DNNOPT over Timeloop (BS2). The time to solution has improved between 35.4× to 6196× across the benchmarks and on average by 249×. Unfortunately the state of the art mapping tool CoSA [26] currently does not support 2D systolic arrays. However, based on their published results [26] we can make a rough indirect estimate. CoSA [26] reports that is on average 1.1× faster than Timeloop random search search where as our approach is 249× faster than Timeloop random search. Therefore our approach would 

DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators 

CF ’24, May 7–9, 2024, Ischia, Italy 

**Figure 10: Improvement Factor of Time to Solution** 

be approximately 249/1.1=226× faster compared to CoSA [26]. Further both CoSA and DNNOPT guarantee optimality. 

## **5.6 Sensitivity Analysis** 

In this section, we question whether the default size of the onchip memory used in previous experiments is a good tradeoff for performance and energy efficiency. To this end, the top diagram of Figure 11 shows the energy consumption for the benchmarks on the baseline architecture relative to the baseline architecture with the default on-chip memory size as a function of the on-chip memory size. The bottom diagram shows the execution time relative to the baseline architecture with the default on-chip memory size as a function of the on-chip memory size. Lower is the better in both cases. We consider six sizes of the on-chip memory where we show the sizes of the input, weight and the output partitions. For example, the smallest on-chip memory uses 64 KB, 32 KB, and 1 KB for input, weights, and output, respectively. 

Considering the normalized execution time first (the bottom diagram), we can see that the execution time drops drastically when we go from the smallest (leftmost) on-chip memory to the one we have picked by default (256 KB + 128 KB + 1 KB). However, while the execution time continues to drop when considering larger on-chip memory, the drop is not so dramatic. 

Moving on to normalized energy consumption as a function of the on-chip memory size, the trend is similar going from the smallest on-chip memory size to the default size. However, when considering larger on-chip memory than the default size, we can see that energy consumption goes up. This is to be expected as the static energy will go up with the size of the on-chip memory. In addition, the area consumed by the on-chip memory will also go up. We conclude that the default size of the on-chip memory is a sound tradeoff between performance and energy efficiency. 

Finally, it is important to understand what the impact of adding the hardware support needed for the compiler methods with respect to the area and power consumption. The MDT is the extra hardware 

**Figure 11: Sensitivity to on-chip memory size. Energy consumption (top) Execution time (bottom).** 

needed. We compare the added area and power consumption of MDT with the default size of on-chip memory. We use CACTI [38] and Accelergy [59] to estimate the area and static power consumption. We find that the area of the MDT is only 0.6% of the the default on-chip memory. The relative ratio of static power consumption is in the same ballpark (0.05 mW vs. 0.00027 mW for on-chip memory vs. MDT, respectively). However, we also compare the dynamic power consumption of the MDT with the default size of the onchip memory. Interestingly, the MDT is accessed only once per layer whereas the on-chip memory is accessed substantially more frequently per layer. The overall effect is that the MDT consumes 1 _._ 4 × 10[−][14] of the power consumed by the on-chip memory. Overall, the added hardware support has a negligible effect on area and power consumption. 

## **6 RELATED WORK** 

Loop reordering and blocking are established techniques that have been used in general-purpose (GP) computing for deep learning [2– 11, 13, 14, 21, 22, 31, 33, 36, 37, 49, 52, 54, 55, 62] to improve cache reuse. However, our focus is on DNN accelerators. [2, 16, 17, 20, 26, 26–30, 34, 35, 35, 35, 39, 41–43, 48, 50, 51, 53, 57, 60, 61, 61, 63, 64] address loop reordering and blocking for DNN accelerators. Typically, a search space with all possible loop orders and blocking factors is established. [26, 34, 35, 39, 41, 43, 53, 57, 60, 61] rely on forms of exhaustively navigate through the search space. While 

CF ’24, May 7–9, 2024, Ischia, Italy 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

they guarantee optimality, the computational complexity is high. In contrast, we have shown that DNNOPT can prune the search space by more than 99% and yet can guarantee optimal results. [16, 20, 26, 35, 51] propose ways to reduce the search space. However, given that all possible sizes for blocking factors and all their combinations should be considered, which some of them ignore, the actual search space still remains large and intractable. 

Prior art has also considered feedback-driven approaches to cut down the search space including black-box auto-tuning, beam search, Markov chain Monte Carlo, and other machine learningbased algorithms with iterative sampling [2, 17, 27, 42, 61]. Specifically, DNN compilers such as TVM [17], TC [32] and XLA [46] support loop optimizations by using auto tuning and ML-aided techniques. Unfortunately, this incurs high training cost. Furthermore, these approaches necessitate massive training datasets and large-scale simulations to learn performance models. Unlike the blackbox statistical approach they use, DNNOPT uses accurate and computationally inexpensive analytical models of data reuse. 

As an alternative for exhaustive search which has high computational complexity, Timeloop [24, 39] proposes random search and [28, 63] use genetic algorithms. A major drawback with the above approaches is that they are not optimal. We have shown that DNNOPT is optimal and yields substantially better results compared to Timeloop random search [39]. 

When considering the computational complexity of evaluating a given point in search space, Timeloop [24, 39] uses consecutive tile analysis and Rahman et al. [43] use simulations. These approaches have a computational complexity of _𝑂_ ( _𝑀_ ). Here, _𝑀_ in Timeloop [39] it is on average 0.3 millions across the layers of Resnet. 

Putra et al. [41] propose analytical cost models for blocking but it ignores temporal reuse of blocks. Other works [20, 35, 40, 50, 51, 64] rely on the loop invariance principle to determine reuse. In contrast, DNNOPT models locality using global reuse distance and interblock reuse. We propose such analytical cost models, for the first time, for modeling a systolic array. 

## **7 CONCLUSION** 

This paper presents DNNOPT, a hardware/software framework for systolic array-based accelerators. It selects loop order and blocking factors, layer-by-layer, to improve the performance and energyefficiency of DNN accelerators by reducing the number of off-chip memory accesses. 

Unlike previous work, DNNOPT selects an optimization _optimally_ and with substantially lower computational complexity. It does so using three strategies: _Early exit_ , _Strided search_ and with the use of _simple and computationally inexpensive analytical cost models_ based on global reuse distance and inter-block reuse. 

However, the key limitation of DNNOPT is that it’s specific to output stationary 2D systolic arrays whereas Timeloop and CoSA could be used for a wider class of DNN accelerators. However, a similar approach as DNNOPT could be used for other classes of accelerators. 

Finally, we evaluate the merits of DNNOPT. DNNOPT prunes the search space by more than two orders of magnitude for CNN workloads. Further, due to its optimal nature, DNNOPT offers 1 _._ 8×, 50% 

and 226× improvement in performance, energy efficiency and time to solution, respectively, compared to state-of-the-art frameworks for the CNN and Transformer benchmarks. 

## **8 ACKNOWLEDGMENT** 

This work was supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and the PRIDE project supported by the Swedish Foundation of Strategic Research under the CHI19-0048 contract. The simulations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725. 

## **REFERENCES** 

- [1] 2023. DeepBench. https://github.com/baidu-research/DeepBench#inferencebenchmark 

- [2] Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li, Michaël Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Durand, et al. 2019. Learning to optimize halide with tree search and random programs. _ACM Transactions on Graphics (TOG)_ 38, 4 (2019), 1–12. 

- [3] Alfred V Aho, Monica S Lam, Ravi Sethi, and Jeffrey D Ullman. 2007. _Compilers: Principles, Techniques, & Tools_ . Pearson Education India. 

- [4] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan-Kelley, Jeffrey Bosboom, Una-May O’Reilly, and Saman Amarasinghe. 2014. Opentuner: An extensible framework for program autotuning. In _Proceedings of the 23rd international conference on Parallel architectures and compilation_ . 303–316. 

- [5] David F Bacon, Susan L Graham, and Oliver J Sharp. 1994. Compiler Transformations for High-performance Computing. _ACM Computing Surveys (CSUR)_ 26, 4 (1994), 345–420. 

- [6] Riyadh Baghdadi, Abdelkader Nadir Debbagh, Kamel Abdous, Benhamida Fatima Zohra, Alex Renda, Jonathan Elliott Frankle, Michael Carbin, and Saman Amarasinghe. [n. d.]. TIRAMISU: A Polyhedral Compiler for Dense and Sparse Deep Learning. ([n. d.]). 

- [7] Riyadh Baghdadi, Jessica Ray, Malek Ben Romdhane, Emanuele Del Sozzo, Abdurrahman Akkas, Yunming Zhang, Patricia Suriana, Shoaib Kamil, and Saman Amarasinghe. 2019. Tiramisu: A polyhedral compiler for expressing fast and portable code. In _2019 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)_ . IEEE, 193–205. 

- [8] Uday Bondhugula, Aravind Acharya, and Albert Cohen. 2016. The pluto+ algorithm: A practical approach for parallelization and locality optimization of affine loop nests. _ACM Transactions on Programming Languages and Systems (TOPLAS)_ 38, 3 (2016), 1–32. 

- [9] Uday Bondhugula, Aravind Acharya, and Albert Cohen. 2016. The pluto+ algorithm: A practical approach for parallelization and locality optimization of affine loop nests. _ACM Transactions on Programming Languages and Systems (TOPLAS)_ 38, 3 (2016), 1–32. 

- [10] Uday Bondhugula, Albert Hartono, Jagannathan Ramanujam, and Ponnuswamy Sadayappan. 2008. A Practical Automatic Polyhedral Parallelizer and Locality Optimizer. In _Proceedings of the 29th ACM SIGPLAN Conference on Programming Language Design and Implementation_ . 101–113. 

- [11] Uday Bondhugula, Albert Hartono, Jagannathan Ramanujam, and Ponnuswamy Sadayappan. 2008. A practical automatic polyhedral parallelizer and locality optimizer. In _Proceedings of the 29th ACM SIGPLAN Conference on Programming Language Design and Implementation_ . 101–113. 

- [12] Amirali Boroumand, Saugata Ghose, Berkin Akin, Ravi Narayanaswami, Geraldo F Oliveira, Xiaoyu Ma, Eric Shiu, and Onur Mutlu. 2021. Google Neural Network Models for Edge Devices: Analyzing and Mitigating Machine Learning Inference Bottlenecks. In _2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT)_ . IEEE, 159–172. 

- [13] Alessio Burrello, Angelo Garofalo, Nazareno Bruschi, Giuseppe Tagliavini, Davide Rossi, and Francesco Conti. 2021. Dory: Automatic end-to-end deployment of real-world dnns on low-cost iot mcus. _IEEE Trans. Comput._ 70, 8 (2021), 1253– 1268. 

- [14] Steve Carr, Kathryn S McKinley, and Chau-Wen Tseng. 1994. Compiler Optimizations for Improving Data Locality. _ACM SIGPLAN Notices_ 29, 11 (1994), 252–262. 

- [15] Daniel Padilla Carrasco, Hatem A Rashwan, Miguel Ángel García, and Domènec Puig. 2021. T-YOLO: Tiny Vehicle Detection Based on YOLO and Multi-Scale Convolutional Neural Networks. _IEEE Access_ (2021). 

- [16] Prasanth Chatarasi, Hyoukjun Kwon, Angshuman Parashar, Michael Pellauer, Tushar Krishna, and Vivek Sarkar. 2021. Marvel: A Data-centric Approach for 

DNNOPT: A Framework for Efficiently Selecting On-chip Memory Loop Optimizations of DNN Accelerators 

CF ’24, May 7–9, 2024, Ischia, Italy 

Mapping Deep Learning Operators on Spatial Accelerators. _ACM Transactions on Architecture and Code Optimization (TACO)_ 19, 1 (2021), 1–26. 

- [17] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al. 2018. {TVM}: An Automated {End-to-End} Optimizing Compiler for Deep Learning. In _13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)_ . 578–594. 

- [18] Yu-Hsin Chen, Joel Emer, and Vivienne Sze. 2016. Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks. _ACM SIGARCH Computer Architecture News_ 44, 3 (2016), 367–379. 

- [19] Yu-Hsin Chen, Tien-Ju Yang, Joel Emer, and Vivienne Sze. 2019. Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices. _IEEE Journal on Emerging and Selected Topics in Circuits and Systems_ 9, 2 (2019), 292–308. 

- [20] Shail Dave, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, and Aviral Shrivastava. 2019. Dmazerunner: Executing Perfectly Nested Loops on Dataflow Accelerators. _ACM Transactions on Embedded Computing Systems (TECS)_ 18, 5s (2019), 1–27. 

- [21] Tobias Grosser, Hongbin Zheng, Raghesh Aloor, Andreas Simbürger, Armin Größlinger, and Louis-Noël Pouchet. 2011. Polly-Polyhedral optimization in LLVM. In _Proceedings of the First International Workshop on Polyhedral Compilation Techniques (IMPACT)_ , Vol. 2011. 1. 

- [22] Tobias Grosser, Hongbin Zheng, Raghesh Aloor, Andreas Simbürger, Armin Größlinger, and Louis-Noël Pouchet. 2011. Polly-Polyhedral optimization in LLVM. In _Proceedings of the First International Workshop on Polyhedral Compilation Techniques (IMPACT)_ , Vol. 2011. 1. 

- [23] Mehmet Günel. 2016. GoogLeNet. 

- [24] Mark Horeni, Pooria Taheri, Po-An Tsai, Angshuman Parashar, Joel Emer, and Siddharth Joshi. 2022. Ruby: Improving hardware efficiency for tensor algebra accelerators through imperfect factorization. In _2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, 254–266. 

- [25] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. 2017. Mobilenets: Efficient Convolutional Neural Networks for Mobile Vision Applications. _arXiv preprint arXiv:1704.04861_ (2017). 

- [26] Qijing Huang, Minwoo Kang, Grace Dinh, Thomas Norell, Aravind Kalaiah, James Demmel, John Wawrzynek, and Yakun Sophia Shao. 2021. Cosa: Scheduling by constrained optimization for spatial accelerators. In _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 554–566. 

- [27] Zhihao Jia, Matei Zaharia, and Alex Aiken. 2019. Beyond Data and Model Parallelism for Deep Neural Networks. _Proceedings of Machine Learning and Systems_ 1 (2019), 1–13. 

- [28] Sheng-Chun Kao and Tushar Krishna. 2020. Gamma: Automating the hw mapping of dnn models on accelerators via genetic algorithm. In _2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD)_ . IEEE, 1–9. 

- [29] Hyoukjun Kwon, Prasanth Chatarasi, Vivek Sarkar, Tushar Krishna, Michael Pellauer, and Angshuman Parashar. 2020. Maestro: A Data-Centric Approach to Understand Reuse, Performance, and Hardware cost of DNN Mappings. _IEEE Micro_ 40, 3 (2020), 20–29. 

- [30] Hyoukjun Kwon and Tushar Krishna. 2018. MAESTRO: An Open-Source Infrastructure for the Cost-Benefit Analysis of Dataflows within Deep Learning Accelerators. (2018). 

- [31] Monica S Lam and Michael E Wolf. 2004. A Data Locality Optimizing Algorithm. _ACM SIGPLAN Notices_ 39, 4 (2004), 442–459. 

- [32] Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. 2020. The deep learning compiler: A comprehensive survey. _IEEE Transactions on Parallel and Distributed Systems_ 32, 3 (2020), 708–727. 

- [33] Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, and P Sadayappan. 2021. Analytical Characterization and Design Space Exploration for Optimization of CNNs. In _Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ . 928–942. 

- [34] Yufei Ma, Yu Cao, Sarma Vrudhula, and Jae-sun Seo. 2017. Optimizing Loop Operation and Dataflow in FPGA Acceleration of Deep Convolutional Neural Networks. In _Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays_ . 45–54. 

- [35] Linyan Mei, Pouya Houshmand, Vikram Jain, Sebastian Giraldo, and Marian Verhelst. 2021. ZigZag: Enlarging Joint Architecture-mapping Design Space Exploration for DNN Accelerators. _IEEE Trans. Comput._ 70, 8 (2021), 1160–1174. 

- [36] Lina Mezdour, Khadidja Kadem, Massinissa Merouani, Amina Selma Haichour, Saman Amarasinghe, and Riyadh Baghdadi. 2023. A Deep Learning Model for Loop Interchange. In _Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction_ . 50–60. 

- [37] Ravi Teja Mullapudi, Andrew Adams, Dillon Sharlet, Jonathan Ragan-Kelley, and Kayvon Fatahalian. 2016. Automatically scheduling halide image processing pipelines. _ACM Transactions on Graphics (TOG)_ 35, 4 (2016), 1–11. 

- [38] Naveen Muralimanohar, Rajeev Balasubramonian, and Norman P Jouppi. 2009. CACTI 6.0: A Tool to Model Large Caches. _HP Laboratories_ 27 (2009), 28. 

- [39] Angshuman Parashar, Priyanka Raina, Yakun Sophia Shao, Yu-Hsin Chen, Victor A Ying, Anurag Mukkara, Rangharajan Venkatesan, Brucek Khailany, Stephen W Keckler, and Joel Emer. 2019. Timeloop: A systematic Approach to DNN Accelerator Evaluation. In _2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, 304–315. 

- [40] Maurice Peemen, Bart Mesman, and Henk Corporaal. 2015. Inter-tile reuse optimization applied to bandwidth constrained embedded accelerators. In _2015 Design, Automation & Test in Europe Conference & Exhibition (DATE)_ . IEEE, 169– 174. 

- [41] Rachmad Vidya Wicaksana Putra, Muhammad Abdullah Hanif, and Muhammad Shafique. 2021. Romanet: Fine-grained Reuse-driven Off-chip Memory Access Management and Data Organization for Deep Neural Network Accelerators. _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ 29, 4 (2021), 702–715. 

- [42] Jonathan Ragan-Kelley, Andrew Adams, Dillon Sharlet, Connelly Barnes, Sylvain Paris, Marc Levoy, Saman Amarasinghe, and Frédo Durand. 2017. Halide: Decoupling Algorithms From Schedules for High-performance Image Processing. _Commun. ACM_ 61, 1 (2017), 106–115. 

- [43] Atul Rahman, Sangyun Oh, Jongeun Lee, and Kiyoung Choi. 2017. Design Space Exploration of FPGA Accelerators for Convolutional Neural Networks. In _Design, Automation & Test in Europe Conference & Exhibition (DATE), 2017_ . IEEE, 1147– 1152. 

- [44] Vijay Janapa Reddi, Christine Cheng, David Kanter, Peter Mattson, Guenther Schmuelling, Carole-Jean Wu, Brian Anderson, Maximilien Breughe, Mark Charlebois, William Chou, et al. 2020. Mlperf Inference Benchmark. In _2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 446–459. 

- [45] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. [n. d.]. Faster R-CNN: Towards Real-Time Object Detection With Region Proposal Networks. _Advances in Neural Information Processing Systems_ 28 ([n. d.]). 

- [46] Amit Sabne. 2020. Xla: Compiling machine learning for peak performance. (2020). 

- [47] Ananda Samajdar, Jan Moritz Joseph, Yuhao Zhu, Paul Whatmough, Matthew Mattina, and Tushar Krishna. 2020. A Systematic Methodology for Characterizing Scalability of DNN Accelerators Using Scale-Sim. In _2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, 58–68. 

- [48] Kevin Siu, Dylan Malone Stuart, Mostafa Mahmoud, and Andreas Moshovos. 2018. Memory Requirements for Convolutional Neural Network Hardware Accelerators. In _2018 IEEE International Symposium on Workload Characterization (IISWC)_ . IEEE, 111–121. 

- [49] Rafael Sousa, Marcio Pereira, Yongin Kwon, Taeho Kim, Namsoon Jung, Chang Soo Kim, Michael Frank, and Guido Araujo. 2023. Tensor slicing and optimization for multicore NPUs. _J. Parallel and Distrib. Comput._ 175 (2023), 66–79. 

- [50] Arthur Stoutchinin, Francesco Conti, and Luca Benini. 2019. Optimally Scheduling CNN Convolutions for Efficient Memory Access. _arXiv preprint arXiv:1902.01492_ (2019). 

- [51] Arne Symons, Linyan Mei, and Marian Verhelst. 2021. Loma: Fast Autoscheduling on DNN Accelerators Through Loop-Order-based Memory Allocation. In _2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS)_ . IEEE, 1–4. 

- [52] Sanket Tavarageri, Gagandeep Goyal, Sasikanth Avancha, Bharat Kaul, and Ramakrishna Upadrasta. 2021. AI Powered Compiler Techniques for DL Code Optimization. _arXiv preprint arXiv:2104.05573_ (2021). 

- [53] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In _Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages_ . 10–19. 

- [54] Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S Moses, Sven Verdoolaege, Andrew Adams, and Albert Cohen. 2018. Tensor comprehensions: Framework-agnostic high-performance machine learning abstractions. _arXiv preprint arXiv:1802.04730_ (2018). 

- [55] Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S Moses, Sven Verdoolaege, Andrew Adams, and Albert Cohen. 2018. Tensor comprehensions: Framework-agnostic high-performance machine learning abstractions. _arXiv preprint arXiv:1802.04730_ (2018). 

- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ 30 (2017). 

- [57] Rangharajan Venkatesan, Yakun Sophia Shao, Miaorong Wang, Jason Clemons, Steve Dai, Matthew Fojtik, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, et al. 2019. Magnet: A modular accelerator generator for neural networks. In _2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)_ . IEEE, 1–8. 

- [58] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang. 2017. Residual Attention Network for Image Classification. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ . 3156–3164. 

Piyumal Ranawaka, Muhammad Waqar Azhar, and Per Stenstrom 

CF ’24, May 7–9, 2024, Ischia, Italy 

- [59] Yannan Nellie Wu, Joel S Emer, and Vivienne Sze. 2019. Accelergy: An architecture-level energy estimation methodology for accelerator designs. In _2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)_ . IEEE, 1–8. 

- [60] Kaiyi Yang, Shihao Wang, Jianbin Zhou, and Takeshi Yoshimura. 2017. Energyefficient Scheduling Method with Cross-loop Model for Resource-limited CNN Accelerator Designs. In _2017 IEEE International Symposium on Circuits and Systems (ISCAS)_ . IEEE, 1–4. 

- [61] Xuan Yang, Mingyu Gao, Qiaoyi Liu, Jeff Setter, Jing Pu, Ankita Nayak, Steven Bell, Kaidi Cao, Heonjae Ha, Priyanka Raina, et al. 2020. Interstellar: Using Halide’s Scheduling Language to Analyze DNN Accelerators. In _Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming_ 

   - _Languages and Operating Systems_ . 369–383. 

- [62] Xuan Yang, Jing Pu, Blaine Burton Rister, Nikhil Bhagdikar, Stephen Richardson, Shahar Kvatinsky, Jonathan Ragan-Kelley, Ardavan Pedram, and Mark Horowitz. 2016. A Systematic Approach to Blocking Convolutional Neural Networks. _arXiv preprint arXiv:1606.04209_ (2016). 

- [63] Ye Yu, Yingmin Li, Shuai Che, Niraj K Jha, and Weifeng Zhang. 2020. Softwaredefined Design Space Exploration for an Efficient DNN Accelerator Architecture. _IEEE Trans. Comput._ 70, 1 (2020), 45–56. 

- [64] Chen Zhang, Peng Li, Guangyu Sun, Yijin Guan, Bingjun Xiao, and Jason Cong. 2015. Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks. In _Proceedings of the 2015 ACM/SIGDA International Symposium on Field-programmable Gate Arrays_ . 161–170. 

