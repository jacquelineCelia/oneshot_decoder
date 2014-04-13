Description of the experiment:

Given a keyword A and spoken examples B1, B2, ..., B20, find B* in B1~B20 that is the closest to A.
===================================================================================================

Input of a command like this:

./gibbs -m 3 -example_index example/phn_files/3_10_1.norm.phn -ex_data example/raw_files/3_10_1.jnas.raw -query_list example/query.2.list -out example/output_files/3_10_1.jnas.out -config example/config_sparse_timit -b ./ -snapshot example/snapshot

-example_index ==> land mark files for the given keyword A. The easiest thing to do here is to create a landmark for each frame.
-ex_data       ==> headerless mfcc feature files for the given Keyword A.
-query_list    ==> paths to the samples that A is going to be compared to, i.e., B1 ~ B20.
-out           ==> path to the output file that saves the result
-config        ==> configuration file that used to train the model
-b ./          ==> (not used)
-snapshot      ==> model file

In query.*.list:
1st col ==> land mark files for B1~B20
2nd col ==> raw mfcc feature data for B1~B20
3rd col ==> used to extract the name, e.g., 3_12_1, for each example. 


The output file:
1st line         ==> path of the output file
2nd line         ==> decoded unit sequence of A
every line after ==> the file name of compared example and the score (the larger the better)
