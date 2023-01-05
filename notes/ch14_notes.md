Original Sentence: 
"I prefer the morning flight through Denver"

Finding the root in a dependency parse is usually trivial. In a dependency parse, the root is the node that is connected to all of the other nodes in the parse tree and does not have any other node connected to it. In the dependency parse you provided, the root is the word "prefer", as it is connected to all of the other words in the parse and does not have any other word connected to it. Here is a tabulation of how each word in the dependency parse is connected:

```
prefer (root)
|
+-- nsubj -- I
|
+-- dobj -- flight
| |
| +-- det -- the
| |
| +-- compound -- morning
|
+-- nmod -- Denver
|
+-- case -- through
```
