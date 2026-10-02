# aralstm

- a multi-stacked bidirectional lstm for arabidopsis genome
- multiclass labelling
- bi-directional lstm with dropout and attention

```
python aralstm.py train sequences.fasta
python aralstm.py train sequences.fasta -k 8 --epochs 50 --batch-size 8 --dropout 0.4

```

```
python aralstm.py predict --seq ACGTACGTTGCAACGTTGCATTAAA --seq GGATCCAAGGCTGGATCCTTGGAAA
python aralstm.py predict --seq-file new_seqs.txt --model my_model.keras

```


Gaurav Sablok \
gsablok@proton.me
