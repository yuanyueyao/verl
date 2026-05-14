<!-- image-->  
(a) Hybrid vs. homogeneous setup performance over training.

<!-- image-->

<!-- image-->  
(b) Response length and epistemic token usage in the homogeneous setup.

Figure 15: Training dynamics of OPSD hybrid distillation in Qwen3-1.7B compared to our homogeneous chat template setup. (a) Under the hybrid setup, the thinking-enabled teacher initially improves student performance, but gains reverse over time. In contrast, the homogeneous setup shows a consistent decrease. (b) Response length and epistemic token usage in the homogeneous setup.

teacher drives the student toward longer responses with improved performance, demonstrating the effectiveness of hybrid distillation in the early phase of training. However, as training progresses, the response length gradually decreases, accompanied by a corresponding degradation in performance. In contrast, under our homogeneous setup as in Figure 15b, where both the student and the teacher have thinking mode enabled, performance declines consistently, while response length and epistemic token count also decrease steadily, consistent with our earlier analysis.

While this hybrid distillation setup for the Qwen3 series presents an interesting research direction with its own unique training dynamics, such as why performance initially improves before declining and whether this stems from changes in reasoning behavior or chat template mismatch, a thorough investigation is beyond the scope of this work and is left for future exploration.

## F More On-Policy Self-Distillation Results

## F.1 Qwen3-8B (Thinking Mode: OFF)

<!-- image-->

<!-- image-->  
(a) Training score-length comparison

<!-- image-->

<!-- image-->  
(b) AMC23 score and response length

<!-- image-->

(c) AIME24 score and response length  
<!-- image-->

<!-- image-->  
(d) Change in epistemic token usage on AIME24

Figure 16: Extended results for Qwen3-8B (thinking mode off): training score-length tradeoff, evaluation performance on AMC23 and AIME24, and change in epistemic token usage.

As an extension of Figure 5 in Section 5.3, we additionally compare AMC23 evaluation scores and the change in epistemic token usage on AIME24. As shown in Figure 16, GRPO significantly increases response length and yields substantial performance gains on both benchmarks. In contrast, SDPO exhibits divergent trends: on AMC23, acc@16 increases from

0.67 to 0.73 while reducing response length by roughly half, whereas on AIME24, acc@16 slightly decreases from 0.25 to 0.23, with pass@16 dropping more substantially. Notably on AMC23, SDPO achieves approximately a 6-point improvement with shorter responses, whereas GRPO obtains a much larger improvement of around 36 points at the cost of considerably longer responses. Achieving large performance gains while maintaining a reasonable response length remains an open challenge.

## F.2 Qwen3-1.7B (Thinking Mode: ON)

To investigate whether our findings generalize to smaller models, we conducted the same onpolicy self-distillation experiment on Qwen3-1.7B. As shown in Figure 17, our finding that self-distillation reduces uncertainty and makes outputs more concise, while simultaneously degrading mathematical reasoning performance, remains consistent across model sizes. Notably, the performance degradation is more severe for the 1.7B model (45.9%) compared to Qwen3-8B with Thinking Mode ON (12.1%) and DeepSeek-R1-Distill-Qwen-7B (40.7%). This suggests that the externalization of uncertainty may be particularly important for robust reasoning in smaller models, which possess more limited inherent capabilities.

<!-- image-->

<!-- image-->  
(a) Training score-length comparison

<!-- image-->

<!-- image-->  
(b) AIME24 score and response length

Figure 17: GRPO and SDPO results on Qwen3-1.7B (Thinking Mode: ON): training score and response length, and OOD evaluation on AIME24. Self-distillation consistently reduces response length but leads to a substantial drop in performance, with a more pronounced degradation compared to larger models.

## F.3 Olmo-3-7B-Instruct

In addition to DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B, we further evaluate on-policy self-distillation on OLMo-3-7B-Instruct from a different model family. As shown in Figure 18, consistent with our previous analyses, SDPO also degrades reasoning performance on this model, with OOD evaluation scores falling below those of the base model. This confirms that our findings are not model-dependent but reflect robust characteristics of reasoning behavior across diverse model families.

<!-- image-->

(a) Training score-length comparison  
<!-- image-->

<!-- image-->

<!-- image-->  
(b) AIME24 score and response length

Figure 18: SDPO results on OLMo-3-7B-Instruct: training score and response length, and OOD evaluation on AIME24.

## F.4 Pass@16 Score

In addition to the acc@16 scores for DeepSeek-Distill-7B and Qwen3-8B (thinking mode enabled) presented in Figures 3b, 3c, 4b, and 4c, we report pass@16 scores in Figure 19. The gap between GRPO and SDPO in pass@16 is larger for DeepSeek-Distill-7B than for Qwen3-8B, and more pronounced on the harder benchmark (AIME24) compared to AMC23.

<!-- image-->

<!-- image-->  
(a) DeepSeek-Distill-7B

<!-- image-->

<!-- image-->  
(b) Qwen3-8B (thinking mode enabled)  
Figure 19: Pass@16 on AMC23 and AIME24 for GRPO and SDPO across training steps.

## G More Ablation Study

To examine the effect of various training hyperparameters on self-distillation behavior, we conduct additional experiments by varying the top-k distillation parameter and the learning rate. As shown in Figure 20a, increasing top-k from 100 to 256 yields no significant difference in training dynamics or final performance. Meanwhile, as illustrated in Figure 20b, reducing the learning rate from 1e−5 to 1e−6 merely slows the rate of degradation; the model ultimately converges to the same reasoning behavior.

<!-- image-->  
(a) Effect of top-k distillation (k = 100 vs. k = 256).

<!-- image-->  
(b) Effect of learning rate (1e−5 vs. 1e−6).  
Figure 20: Ablation study on top-k distillation and learning rate. Both modifications fail to prevent the convergence toward degraded reasoning behavior; a lower learning rate only delays the process.

## H Qualitative Analysis

## H.1 Unguided vs. Solution-Guided Generation

We analyze examples of ${ \mathcal { D } } _ { \mathrm { u g } }$ (Unguided responses) and $\mathcal { D } _ { \mathrm { s g } }$ (Solution-guided responses) introduced in Section 4. Below, we present the two responses generated for the same problem. As illustrated, the unguided response exhibits substantially more uncertainty and exploratory reasoning, frequently revisiting assumptions, considering alternative approaches, and self-correcting along the way. In contrast, the solution-guided response proceeds in a more concise and direct manner, arriving at each intermediate result without detours.

Problem. How many ways can 8 mutually non-attacking rooks be placed on a $9 \times 9$ chessboard so that all 8 rooks are on squares of the same color? (Two rooks are said to be attacking each other if they are placed in the same row or column of the board.) Please reason step by step, and put your final answer within

## Unguided Response – Exploratory Reasoning

Alright, so I have this problem where I need to figure out how many ways I can place 8 mutually non-attacking rooks on a 9×9 chessboard, and all of them have to be on squares of the same color. Hmm, let me break this down.

First, I know that a chessboard is 8×8, but this is a 9×9 chessboard. So, it’s one square larger on each side. Also, non-attacking rooks mean that no two rooks can be in the same row or

column. Since it’s 9×9, if I were placing 9 rooks, they would each occupy their own row and   
column. But here, we’re placing 8 rooks, so one row and one column will be empty.   
But wait, the problem also says all rooks must be on squares of the same color. So, not only do   
they have to not attack each other, but they all have to be on, say, white squares or all on black   
squares.   
[. . . ]   
So, perhaps, the number of ways to place 8 non-attacking rooks on white squares would be   
equal to the number of permutation matrices on the white squares.   
Similarly for black squares.   
But how do we compute that?   
I recall that in combinatorics, the number of permutation matrices avoiding certain positions   
is given by the inclusion-exclusion principle, but that might get complicated.   
Alternatively, maybe we can model this as a bipartite graph matching problem.   
[. . . ]   
But I don’t know the exact formula for that.   
Alternatively, perhaps we can use the principle of inclusion-exclusion or recursion.   
Alternatively, perhaps notice that the white squares form a bipartite graph which is a union of   
two complete bipartite graphs, but that might not hold.   
Wait, let me consider the structure of the white squares. [. . . ] So, the bipartite graph is actually   
composed of two complete bipartite graphs: one connecting even rows to even columns, and   
the other connecting odd rows to odd columns.   
Therefore, the bipartite graph is actually two disjoint complete bipartite graphs: one between   
the odd rows (5 rows) and odd columns (5 columns), and another between the even rows (4   
rows) and even columns (4 columns).   
So, the white squares form two separate complete bipartite graphs: $K _ { 5 , 5 }$ and $K _ { 4 , 4 }$   
[. . . ]   
So, to get a matching of size $^ { 8 , }$ we need to have k edges from $K _ { 5 , 5 }$ and $8 - k$ edges from $K _ { 4 , 4 } ,$   
where k ranges from 4 to 5.   
$\operatorname { I f } k = 4 ,$ then $8 - 4 = 4$ edges from $K _ { 4 , 4 }$   
If k = 5, then 8 − 5 = 3 edges from $K _ { 4 , 4 } .$   
[. . . ]   
So, for $K _ { 5 , 5 } ,$ the number of matchings of size 4 is ${ \bigl ( } _ { 4 } ^ { 5 } { \bigr ) } ^ { 2 } \cdot 4 ! = 2 5 \times 2 4 = 6 0 0 .$   
Similarly, the number of matchings of size 5 in $\begin{array} { r } { K _ { 5 , 5 } \mathrm { i s } \left( \frac { 5 } { 5 } \right) ^ { 2 } \cdot 5 ! = 1 \times 1 2 0 = 1 2 0 . } \end{array}$   
For $K _ { 4 , 4 } ,$ the number of matchings of size 4 is ${ \bigl ( } { \begin{array} { l } { 4 } \\ { 4 } \end{array} } ^ { 2 } \cdot 4 ! = 1 \times 2 4 = 2 4 .$   
And the number of matchings of size 3 is ${ \bigl ( } { \begin{array} { l } { 4 } \\ { 3 } \end{array} } ^ { 2 } \cdot 3 ! = 1 6 \times 6 = 9 6 .$   
Therefore: $6 0 0 \times 2 4 + 1 2 0 \times 9 6 = 1 4 , 4 0 0 + 1 1 , 5 2 0 = 2 5 , 9 2 0 .$   
[. . . ]   
In $K _ { 5 , 4 } ,$ the number of perfect matchings (size 4) is $( _ { 4 } ^ { 5 } ) \cdot { \binom { 4 } { 4 } } \cdot 4 ! = 5 \times 1 \times 2 4 = 1 2 0 .$   
Similarly, in $K _ { 4 , 5 } ,$ the number of perfect matchings is $( { \binom { 4 } { 4 } } \cdot { \binom { 5 } { 4 } } \cdot 4 ! = 1 \times 5 \times 2 4 = 1 2 0 .$   
Therefore, the number of matchings of size 8 in the black square bipartite graph is $1 2 0 \times 1 2 0 =$   
14,400.   
Therefore, total is $2 5 , 9 2 0 + 1 4 , 4 0 0 = 4 0 , 3 2 0 .$   
But wait, 40,320 is $8 ! = 4 0 { , } 3 2 0 .$ . Hmm, that’s interesting.   
But why?   
Is there a combinatorial interpretation where the number of ways is 8! multiplied by some  
thing?   
Wait, 8! is 40320, which is exactly the number of ways we got.   
But why is it exactly 8!? It seems coincidental.   
[. . . ]   
Wait, perhaps it’s best to just accept that the total number is 40,320.   
So, the final answer is 40,320

## Solution-Guided Response – Concise, Direct Reasoning

Alright, so I have this problem where I need to figure out how many ways I can place 8 mutually non-attacking rooks on a 9×9 chessboard, and all of them have to be on squares of the same color. Hmm, okay, let’s break this down step by step.

```latex
First, I know that a rook can move any number of squares vertically or horizontally, so non
attacking rooks must be placed on different rows and different columns. That means for 8
rooks, we need 8 distinct rows and 8 distinct columns.
[. . . ] Specifically, there are 41 squares of one color and 40 of the other. $[ \ldots ]$ So white squares
are 41 and black squares are 40.
So, I need to calculate the number of ways to place 8 non-attacking rooks on white squares
and then on black squares separately, and then add them together.
Starting with white squares. $[ . . . ]$ The white square bipartite graph has two components: $K _ { 5 , 5 }$
and $\check { K _ { 4 , 4 } }$ . Similarly, [. . . ] the black square bipartite graph has two components: $\dot { K } _ { 5 , 4 }$ and $K _ { 4 , 5 } .$
[. . . ] The number of ways to choose 4 edges from $K _ { 5 , 5 }$ is ${ \binom { 5 } { 4 } } ^ { 2 } \cdot 4 ! = 2 5 \times 2 4 = 6 0 0$ . The
number of ways to choose 5 edges from $K _ { 5 , 5 } \mathrm { i s } \left( { \frac { 5 } { 5 } } \right) ^ { 2 } \cdot 5 ! = 1 \times 1 2 0 = 1 2 0$ . Similarly, for $K _ { 4 , 4 } ,$
the number of ways to choose 4 edges is ${ \bigl ( } { \frac { 4 } { 4 } } { \bigr ) } ^ { 2 } \cdot 4 ! = 1 \times 2 4 = 2 4 ,$ and choosing 3 edges is
${ \bigl ( } { \frac { 4 } { 3 } } { \bigr ) } ^ { 2 } \cdot 3 ! = 1 6 \times 6 = 9 6 .$
So, the total number of ways for white squares is $( 6 0 0 \times 2 4 ) + ( 1 2 0 \times 9 6 ) = 1 4 { , } 4 0 0 + 1 1 { , } 5 2 0 =$
25,920.
For black squares, $[ \ldots ]$ the total number of ways is 14,400.
Adding both cases together, we get $2 5 , 9 2 0 + 1 4 , 4 0 0 = \left| 4 0 , 3 2 0 \right|$
```

## H.2 GRPO vs. SDPO Generation

Problem (AIME24). Find the number of ways to place a digit in each cell of a $2 \times 3$ grid so that the sum of the two numbers formed by reading left to right is 999, and the sum of the three numbers formed by reading top to bottom is 99. The grid below is an example of such an arrangement because $8 + 9 9 1 ^ { \circ } = { \dot { 9 } } 9 9$ and $9 + 9 + 8 1 = 9 9 \colon$

<table><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td></tr></table>

The correct answer is $\boxed { 4 5 }$

GRPO explores with repeated reflections, deriving the extra constraint $a + b + c = 8$ beyond the pairwise sums, and correctly counts ${ \binom { 1 0 } { 2 } } = 4 5$ solutions. SDPO proceeds directly and stops after deriving $a + d = b + e = c + { \dot { f } } = 9 ,$ , multiplying $1 0 \times 1 0 ^ { - } \times 1 0 = 1 0 0 0$ without checking the row-sum equation, thereby missing the additional constraint $a + b + c = 8$ .

DeepSeek-R1-Distill-Qwen-7B GRPO Generation (Correct: 45)   
Okay, so I need to figure out how many ways there are to fill $\mathsf { a } 2 \times 3$ grid with digits such that   
two conditions are satisfied: (1) the sum of the two numbers formed by reading left to right is   
999, and (2) the sum of the three numbers formed by reading top to bottom is 99.   
Let me visualize the grid as ${ \sqrt { \left. { \frac { a } { d } } \right| \left. { \frac { b } { e } } \right| { \mathcal { I } } \ } } .$ The first condition gives (100a $+ 1 0 b + c ) +$ (100d +   
$1 0 e + f ) = 9 9 9 ,$ , and the second gives $\overline { { ( 1 0 a + d ) } } + ( 1 0 b + e ) + ( 1 0 c + f ) = 9 9 .$   
$[ \ldots ]$   
So we have:   
Equation (A): 100 $S _ { 1 } + 1 0 S _ { 2 } + S _ { 3 } = 9 9 9 ,$ where $S _ { 1 } = a + d , S _ { 2 } = b + e , S _ { 3 } = c + f .$   
Equation (B): $1 0 T + U = 9 9 ,$ where $T = a + b + c , U = d + e + f .$   
$[ . . . ]$ Since $S _ { 1 }$ is $a + d , S _ { 1 }$ can be from 0 to 18. But if $S _ { 1 }$ is $^ { 1 0 , }$ then $1 0 0 S _ { 1 } = 1 0 0 0 ,$ , which is   
already larger than 999. Therefore, $S _ { 1 }$ can be from 0 to 9.   
For $S _ { 1 } \dot { = } 9 \dot { : } 1 0 0 \cdot 9 = 9 0 0$ . Then $1 0 { \dot { S } } _ { 2 } { \dot { + } } S _ { 3 } = 9 9 .$   
For $S _ { 1 } = 8 \colon 1 0 0 \cdot 8 = 8 0 0$ . Then $1 0 S _ { 2 } + S _ { 3 } = 1 9 9$ . But $1 0 S _ { 2 } + S _ { 3 }$ can be at most $1 0 \cdot 1 8 + 1 8 =$   
198. So 199 is too big. Therefore $S _ { 1 }$ can’t be 8.   
For $S _ { 1 } = 9 , 1 0 S _ { 2 } + \breve { S } _ { 3 } = 9 9 .$ . Since $S _ { 2 }$ and $S _ { 3 }$ are between 0 and $1 8 , 1 0 S _ { 2 } + S _ { 3 }$ can be between 0   
and 198. So 99 is possible.   
[. . . ] Therefore $S _ { 2 } ^ { ^ { + } }$ must be 9. Hence $, S _ { 2 } = 9 ,$ then $S _ { 3 } = 9 9 - 9 0 = 9 .$   
Let me check other $S _ { 1 }$ values. For $S _ { 1 } = 7 \colon 1 0 S _ { 2 } +  { S _ { 3 } } = 2 9 9 ,$ , even larger. Hence impossible.   
Similarly for lower $S _ { 1 }$ . Hence, only $S _ { 1 } = 9$ is possible.

Wait, let me double-check that $S _ { 1 } = 9$ is the only possible value. For $S _ { 1 } = 9 ,$ we have   
$1 0 S _ { 2 } + S _ { 3 } = 9 9 _ { , }$ , which requires $S _ { 2 } = 9$ and $S _ { 3 } = 9 . \dot { [ \dots ] }$ So only $S _ { 2 } = 9$ and $S _ { 3 } = 9 .$   
Therefore, Equation (A) only holds when $S _ { 1 } = S _ { 2 } = { S _ { 3 } } = 9 .$   
[. . . ] From $T ^ { \bullet } + U = 2 7$ and $\stackrel { \smile } { 1 0 } T + U = 9 9 :$ subtracting gives $9 T = 7 2 ,$ , so $T = 8 , U = 1 9 .$   
[. . . ] Substituting $d = 9 - a , e = 9 - b , f = 9 - c ,$ Equation $5 \ : ( d + e + f = 1 9 )$ is automatically   
satisfied. The main constraint reduces to a $+ b + c = 8$ with ${ \dot { a } } , b , c \in \{ 0 , \ldots , 9 \}$   
Since $8 < 9 ,$ the upper bound is automatic, so the count is the number of non-negative integer   
solutions to $a + b + c = 8 ,$ which is ${ \binom { 1 0 } { 2 } } = 4 5 .$   
Since $d = 9 - a$ and $a \leq 8 , d$ ranges from 1 to $^ { 9 - }$ all valid digits. Similarly for $e , f .$ Therefore,   
all 45 solutions are valid.   
Wait, let me check if there are constraints I missed. When forming the numbers $N _ { 1 }$ and $N _ { 2 } ,$   
they can have leading zeros, which is fine since the example itself has 008 in the top row. $[ \dots ]$   
So there is no problem with leading zeros.   
Let me verify with the given example: $a = 0 , b = 0 , c = 8 ,$ so $d = 9 , e = 9 , f = 1 .$ Columns:   
$0 9 + 0 9 + 8 1 { \overset { \cdot } { = } } 9 9 { \overset { . } { \surd } }$ . Rows: 008 + 991 = 999 ✓.   
Another example: $a = 8 , b = 0 , c = 0 .$ Then $d = 1 , e = 9 , f = 9 .$ Top: 800, bottom: 199.   
800 + 199 = 999 ✓. Columns: 81 $+ 9 + 9 = 9 9 { \sqrt { \phantom { 9 } } } .$   
Another: $a = 0 , b = 1 , c = 7 .$ Then $d = 9 , e = 8 , f = 2 . 1 7 + 9 8 2 = 9 9 9 \check { \sqrt { . 9 + 1 8 + 7 2 } } = 9 9 \check { \sqrt { . } }$   
Wait, one more check — what if leading zeros aren’t allowed? Then the top row would need   
$a \neq 0 ,$ but the example given has $a = 0 ,$ so that interpretation is invalid. Therefore, leading   
zeros are allowed, and the answer is indeed 45.   
Therefore, I think the correct answer is $\boxed { 4 5 } .$

DeepSeek-R1-Distill-Qwen-7B SDPO Generation (Incorrect: 1000)   
Okay, so I need to figure out how many ways there are to fill a $2 \times 3$ grid with digits so that   
the sum of the two numbers formed by reading left to right is 999, and the sum of the three   
numbers formed by reading top to bottom is 99.   
Labeling the grid as Row 1: ${ \check { a } } , { \check { b } } , c$ and Row $2 \colon d , e , f ,$ the two conditions give:   
(1) $1 0 ( a + b + c ) + ( d + e + f ) = 9 9$   
(2) $1 0 0 ( a + d ) + 1 0 ( b + e ) + ( c + f ) = 9 9 9$   
Let $S = a + b + c , T = d + e + f ,$ and $x = a + d , y = b + e , z = c + f .$ Then (1) becomes   
$1 0 S + T = 9 9$ and (2) becomes $1 { \dot { 0 } } 0 x + 1 0 y + z = 9 9 { \dot { 9 } } .$   
Since x $+ y + z = S + T ,$ , substituting $T = 9 9 - 1 0 S$ gives $x + y + z = 9 9 - 9 S$ . Subtracting   
this from (2):   
$9 9 x + 9 y = 9 0 0 + 9 S \Rightarrow 1 1 x + y = 1 0 0 + S .$   
[. . . ] From $0 \leq 9 9 - 1 0 S \leq 2 7$ we get $S \in \{ 8 , 9 \} .$   
Case 1: $S = 8 .$ Then $T = 1 9 ,$ 11x $+ y = 1 0 8 ,$ and $x + y + z = 2 7 .$ . Substituting $y = 1 0 8 - 1 1 x$   
gives z = 10x − 81. The constraint $0 \le z \le 1 8$ forces $x = 9 ,$ so $y = 9 , z = 9 .$ Therefore   
$a + d = 9 , b + e = 9 , c + f = 9 .$   
For each of the three pairs, there are 10 possibilities (since for each digit from 0 to 9, there’s   
exactly one digit that adds up to 9). So for each of the three pairs, there are 10 possibilities.   
Therefore, the total number of solutions in this case is $1 0 \times 1 0 ^ { - } \times 1 0 ^ { - } = 1 0 0 0 .$   
Case 2: $S = 9 .$ Then $T = 9 , 1 1 x + y = 1 0 9 ,$ , and $x + y + z = 1 8 .$ Substituting gives $z = 1 0 x - 9 1 .$   
The constraint $0 \le z \le 1 8$ forces $x = 1 0 ,$ , but then $y = 1 0 9 - 1 1 0 = - \mathrm { \tilde { 1 } }$ , which is invalid.   
Therefore, this case is invalid.   
Thus, only Case 1 is valid, leading to 1000 possible configurations.   
The digits $a , b , c , d , e , f$ must be digits between 0 and 9. In Case 1, we have $a + d = 9 ,$ so each   
pair $( a , d )$ can be any pair of digits adding to 9, which is valid. Similarly for $( b , e )$ and $( c , f )$   
So there are no invalid digit values here. Therefore, all 1000 configurations are valid.   
Let me check with the example given in the problem: ${ \sqrt { \frac { 0 } { 9 } \int 0 \ { \sqrt { \ 8 } } } } .$ Horizontal sum: 008 +   
$9 9 1 = 9 9 9 { \sqrt { \phantom { 9 9 } } } \times .$ . Vertical sum: $0 9 + 0 9 + 8 1 = 9 9 { \sqrt { \phantom { 0 } } } .$ . Each of the pairs $( a , d ) = ( 0 , 9 ) , ( b , e ) =$   
(0, 9), $( c , f ) = ( 8 , 1 )$ adds to 9, which fits Case 1. So this example is counted in the 1000   
configurations.   
Hence, the answer should be 1000 .