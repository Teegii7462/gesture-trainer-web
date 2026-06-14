/*
 * Random-forest evaluation in the browser. The Python exporter
 * (`gesture_tracker/web_export.py`) flattens each scikit-learn decision tree into
 * parallel arrays with a normalized class distribution per node. Averaging the
 * per-tree leaf distributions reproduces `RandomForestClassifier.predict_proba`
 * exactly, so the website's predictions match the trained model.
 */

/** One decision tree as flat arrays (leaf nodes have left === -1). */
export interface SerializedTree {
  /** Split feature index per node; -2 at leaves. */
  readonly feature: number[];
  /** Split threshold per node (split goes left when x[feature] <= threshold). */
  readonly threshold: number[];
  /** Left child index per node; -1 at leaves. */
  readonly left: number[];
  /** Right child index per node; -1 at leaves. */
  readonly right: number[];
  /** Normalized class distribution per node (only leaves are read). */
  readonly value: number[][];
}

export interface SerializedForest {
  /** Class labels in the order the per-node `value` arrays use. */
  readonly classes: string[];
  readonly n_estimators: number;
  readonly trees: SerializedTree[];
}

/** Walk one tree to its leaf and return that leaf's class distribution. */
function treeLeafValue(tree: SerializedTree, x: number[]): number[] {
  let node = 0;
  // Leaves have left === -1; internal nodes always have both children.
  while (tree.left[node] !== -1) {
    node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
  }
  return tree.value[node];
}

/**
 * Mean of the per-tree leaf distributions = forest class probabilities, in
 * `forest.classes` order.
 */
export function forestProba(forest: SerializedForest, x: number[]): number[] {
  const nClasses = forest.classes.length;
  const acc = new Array<number>(nClasses).fill(0);
  for (const tree of forest.trees) {
    const leaf = treeLeafValue(tree, x);
    for (let c = 0; c < nClasses; c++) acc[c] += leaf[c];
  }
  const n = forest.trees.length;
  for (let c = 0; c < nClasses; c++) acc[c] /= n;
  return acc;
}
