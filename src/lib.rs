use pyo3::prelude::*;
use rustfst::algorithms::compose;
use rustfst::algorithms::tr_sort as rs_tr_sort;
use rustfst::prelude::*;
use std::path::Path;
use std::sync::Arc;
// use rustfst::utils::transducer;
use rustfst::utils::acceptor;
use rustfst::fst_traits::SerializableFst;

/// Wraps a [`rustfst`] SymbolTable struct as a Python class.
///
/// # Example
/// ## Rust
/// ```
/// let symt = SymTab::new(vec!["a", "b", "c"]);
/// assert_eq!(symt.get_symbol(1), "a");
/// ```
/// ## Python
/// ```{.python}
/// symt = SymTab(["a", "b", "c"])
/// assert(symt.get_symbol(1) == "a")
/// ```
#[pyclass]
pub struct SymTab {
    /// A SymbolTable struct from [`rustfst`]
    symt: SymbolTable,
}
#[pymethods]
impl SymTab {
    /// Constructs a new [`SymTab`] instance
    ///
    /// # Example
    ///
    /// ```
    /// let symt = SymTab::new(vec!["a", "b", "c"]);
    /// ```
    #[new]
    pub fn new(sigma: Vec<String>) -> Self {
        let mut symt = SymbolTable::new();
        for s in sigma {
            symt.add_symbol(s);
        }
        SymTab { symt: symt }
    }

    /// Adds a symbol to a [`SymTab`] with an associated label
    ///
    /// # Example
    ///
    /// ```
    /// let symt = SymTab::new(vec!["a", "b"]);
    /// symt.add_symbol("c");
    /// assert_eq!(symt.get_symbol(3)?, "c");
    /// ```
    pub fn add_symbol(&mut self, s: &str) -> PyResult<()> {
        self.symt.add_symbol(s);
        Ok(())
    }

    /// Given a symbol, returns the corresponding label
    ///
    /// # Example
    ///
    /// ```
    /// let symt = SymTab::new(vec!["a", "b", "c"]);
    /// assert_eq!(symt.get_label("b")?, 2);
    /// ```
    pub fn get_label(&self, s: &str) -> Option<Label> {
        self.symt.get_label(s)
    }

    /// Given a label, returns the corresponding symbol
    ///
    /// # Example
    ///
    /// ```
    /// let symt = SymTab::new(vec!["a", "b", "c"]);
    /// assert_eq!(symt.get_symbol(3), "c");
    /// ```
    pub fn get_symbol(&self, l: Label) -> Option<&str> {
        self.symt.get_symbol(l)
    }
}

/// Wraps the [`VectorFst`] struct from [`rustfst`]. Assumes weights are in the tropical semiring.
/// 
/// # Example
/// ## Rust
/// 
/// ```
/// let t = WeightedFst::new();
/// ```
/// use wfst4str::WeightedFst;
/// 
/// let t = WeightedFst();
/// let sym = vec!["a", "b", "c"];
/// t.set_input_symbols(sym);
/// t.set_output_symbols(sym);
/// let q0 = t.add_state();
/// let q1 = t.add_state();
/// t.set_start(q0).unwrap();
/// t.set_final(q1).unwrap();
/// t.add_tr(0, 1, "a", "b").unwrap();
/// ```
/// ## Python
/// 
/// ```python
/// import wfst4str
/// 
/// t = wfst4str.WeightedFst();
/// t.set_input_symbols(['a', 'b', 'c'])
/// t.set_output_symbols(['a', 'b', 'c'])
/// q0 = t.add_state()
/// q1 = t.add_state()
/// t.set_start(q0)
/// q1.set_final(q1, 0.0)
/// t.add_tr(0, 1, 'a', 'b')
/// ```
/// 
#[pyclass]
pub struct WeightedFst {
    fst: VectorFst<TropicalWeight>,
}

#[pymethods]
impl WeightedFst {

    /// Constructs a [`WeightedFst`] object.
    #[new]
    pub fn new() -> Self {
        WeightedFst {
            fst: VectorFst::new(),
        }
    }

    /// Creates a symbol table from a vector of strings and associates it with the wFST. 
    pub fn set_input_symbols(&mut self, sym_vec: Vec<String>) -> () {
        let symt = SymTab::new(sym_vec);
        self.fst.set_input_symbols(Arc::new(symt.symt))
    }

    /// Creates a symbol table from a vector of strings and associates it with the wFST.
    pub fn set_output_symbols(&mut self, sym_vec: Vec<String>) -> () {
        let symt = SymTab::new(sym_vec);
        self.fst.set_output_symbols(Arc::new(symt.symt))
    }

    /// Adds `n` states to the wFST.
    pub fn add_states(&mut self, n: usize) {
        self.fst.add_states(n)
    }

    /// Adds one new state and returns the corresponding number. 
    pub fn add_state(&mut self) -> StateId {
        self.fst.add_state()
    }

    /// Sets `state` as a start state (initial state).
    pub fn set_start(&mut self, state: StateId) -> PyResult<()> {
        self.fst
            .set_start(state)
            .unwrap_or_else(|e| panic!("Cannot set {:?} as start state: {}", state, e));
        Ok(())
    }

    /// Sets `state` as a final state (accepting state).
    pub fn set_final(&mut self, state: StateId, weight: f32) -> PyResult<()> {
        self.fst
            .set_final(state, weight)
            .unwrap_or_else(|e| panic!("Cannot set {:?} as final state: {}", state, e));
        Ok(())
    }

    /// Serializes the wFST to a text file at `path` in AT&T format (OpenFST compatible).
    pub fn write_text(&self, path: &str) -> PyResult<()> {
        let path_output = Path::new(path);
        self.fst.write_text(path_output).unwrap();
        Ok(())
    }

    /// Returns a serialization of the wFST as a string in AT&T format (OpenFST compatible). 
    pub fn text(&self) -> String {
        match self.fst.text() {
            Ok(s) => s,
            Err(s) => panic!("Cannot produce text representation: {}", s),
        }
    }

    /// Outputs a representation of the wFST as a `dot` file to `path`, which
    /// can be visualized with Graphviz.
    pub fn draw(&self, path: &str) -> PyResult<()> {
        let config = DrawingConfig::default();
        let path_output = Path::new(path);
        self.fst
            .draw(path_output, &config)
            .unwrap_or_else(|e| panic!("Cannot write to path {}: {}", path, e));
        Ok(())
    }

    /// Adds a transition to the wFST from `source_state` to `next_state` with
    /// input symbol `isym`, output symbol `osym`, and weight `weight`
    pub fn add_tr(
        &mut self,
        source_state: StateId,
        next_state: StateId,
        isym: &str,
        osym: &str,
        weight: f32,
    ) -> PyResult<()> {
        let empty_symt = Arc::new(SymbolTable::new());
        let ilabel = &self
            .fst
            .input_symbols()
            .unwrap_or(&empty_symt)
            .get_label(isym)
            .unwrap_or(0);
        let olabel = &self
            .fst
            .output_symbols()
            .unwrap_or(&empty_symt)
            .get_label(osym)
            .unwrap_or(0);
        let tr = Tr::<TropicalWeight>::new(*ilabel, *olabel, weight, next_state);
        self.fst.add_tr(source_state, tr).unwrap();
        Ok(())
    }

    /// Returns the composition of the wFST and another wFST (`other`)
    pub fn compose(&self, other: &WeightedFst) -> PyResult<WeightedFst> {
        Ok(WeightedFst {
            fst: compose::compose(self.fst.clone(), other.fst.clone()).expect("Couldn't compose!"),
        })
    }

    /// Converts a string to a linear wFST using the input `SymbolTable` of the wFST.
    pub fn to_linear_fst(&self, s: &str) -> PyResult<WeightedFst> {
        let symt = self
            .fst
            .input_symbols()
            .unwrap();
        let mut syms: Vec<String> = Vec::new();
        let mut acc: String = String::from("");
        for c in s.chars() {
            acc.push(c);
            if !symt.contains_symbol(&acc) {
                match acc.pop() {
                    Some(n) => {
                        syms.push(acc);
                        acc = n.to_string();
                    }
                    None => (),
                }
            }
        }
        syms.push(acc);
        fn semiring_one<W: Semiring>() -> W {
            W::one()
        }
        let lfst: VectorFst<TropicalWeight> = acceptor(
            &syms.iter().map(|x| symt.get_label(x).unwrap()).collect::<Vec<usize>>()[..],
            semiring_one(),
        );
        println!("In `to_linear_fst`: `lfst.text()=`\n{}", lfst.text().unwrap());
        Ok(WeightedFst { fst: lfst })
    }

    /// Applies the wFST to a string (consisting of symbols in the wFSTs `SymbolTable`s).
    pub fn apply(&self, s: &str) -> PyResult<Vec<String>> {
        let lfst = self.to_linear_fst(s).unwrap();
        println!("lfst.text()=\n{}", lfst.text());
        println!("self.text()=\n{}", self.text());
        let fst2 = lfst.compose(self).unwrap();
        println!("fst2.text()=\n{}", fst2.text());
        let empty_symt = Arc::new(SymbolTable::new());
        println!("{}", fst2.text());
        Ok(fst2
            .fst
            .paths_iter()
            .map(|p| {
                p.olabels
                    .iter()
                    .map(|&l| {
                        self.fst
                            .output_symbols()
                            .unwrap_or(&empty_symt)
                            .get_symbol(l)
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect())
    }

    /// Sorts the transitions of a wFST based on its input labels.
    pub fn tr_ilabel_sort(&mut self) {
        let _comp = ILabelCompare {};
        rs_tr_sort(&mut self.fst, _comp)
    }

    /// Sorts the transitions of a wFST based on its output labels.
    pub fn tr_olabel_sort(&mut self) {
        let _comp = OLabelCompare {};
        rs_tr_sort(&mut self.fst, _comp)
    }
}

// #[pyfunction]
// pub fn compose(fst1: &WeightedFst, fst2: &WeightedFst) -> WeightedFst {
//     match rscompose(fst1.fst.clone(), fst2.fst.clone()) {
//         Ok(fst) => WeightedFst { fst },
//         Err(e) => panic!("Cannot compse WFSTs: {}", e),
//     }
// }

// #[pyfunction]
// pub fn tr_ilabel_sort(fst: &mut WeightedFst) {
//     let _comp = ILabelCompare {};
//     rs_tr_sort(&mut fst.fst, _comp)
// }

// #[pyfunction]
// pub fn tr_olabel_sort(fst: &mut WeightedFst) {
//     let _comp = OLabelCompare {};
//     rs_tr_sort(&mut fst.fst, _comp)
// }

#[pyfunction]
pub fn from_text_string(fst_string: &str) -> PyResult<WeightedFst> {
    Ok(WeightedFst { fst: VectorFst::from_text_string(&fst_string).unwrap() })
}

#[pyfunction]
pub fn read_text(path_text_fst: &str) -> PyResult<WeightedFst> {
    let fst_path = Path::new(path_text_fst);
    Ok(WeightedFst { fst: VectorFst::read_text(fst_path).unwrap()})
}

#[pymodule]
#[pyo3(name = "wfst4str")]
fn wfst4str(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SymTab>()?;
    m.add_class::<WeightedFst>()?;
    m.add_function(wrap_pyfunction!(from_text_string, m)?)?;
    m.add_function(wrap_pyfunction!(read_text, m)?)?;
    // m.add_function(wrap_pyfunction!(compose, m)?)?;
    // m.add_function(wrap_pyfunction!(tr_ilabel_sort, m)?)?;
    // m.add_function(wrap_pyfunction!(tr_olabel_sort, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_get_symbol() {
        let st = crate::SymTab::new(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(st.get_symbol(1).unwrap_or(""), "a");
        assert_eq!(st.get_symbol(2).unwrap_or(""), "b");
        return ();
    }

    #[test]
    fn test_apply() {
        ()
    }
}
