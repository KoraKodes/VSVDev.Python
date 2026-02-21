Write python snippets in a jupyter notebook for implementing the following.

The term 'array' and 'table' shall indicate dataframes, numpy arrays, or similar data structures (select the most appropriate one). The tables have labels representing EU countries as row index, years as the column index. The arrays may have the same labels as the row index of the tables.

## Task 1

The first task consists in defining code to generate the array A_i from the table T_i for i=1,2,... The arrays and tables contain numercal data. The arrays A_i is computed from the table T_i by applying a chain of operations. For instance:

A_1 = weighted average of the columns of T_1 for the years 2010-2019, whereby the weights and year selection are parameters of the average (e.g. other arrays W_j)

A_2 = arithermetic average of the columns of T_2 for the years y1, y2,... and then sqrt of the average.

A_3 = log of the column y4 of T_3

In general:

T_i -> operation_1 -> operation_2 -> ... -> operation_n -> A_i

Use where possible matrix operations or syntax (whereby vectors a matrix as well!). MAybe the 'delegate' pattern is useful here, or the monoids.

## Task 2

Let W_i be a weight array for i=1,2,... , and let M_i be a mask array for i=1,2,... . The arrays A_i and M_i have the same length (and labels / indexing); the M_i are however boolean arrays (or 1/0 arrays). The M_i are used to select the elements of A_i to be used in the computation of the W_i. Given the A_l and A_k, the W_l and W_k, build A_x by taking A_l, W_l if M is true, and A_k, W_k if M is false. Again, possibly use matrix operations or syntax, and the delegate pattern or monoids.

