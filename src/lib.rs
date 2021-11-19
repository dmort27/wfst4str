use pyo3::prelude::*;
use rustfst::algorithms::compose;
use rustfst::algorithms::tr_sort as rs_tr_sort;
use rustfst::prelude::*;
use std::path::Path;
use std::sync::Arc;
// use rustfst::utils::transducer;
use rustfst::utils::acceptor;

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

#[pyclass]
pub struct WeightedFst {
    fst: VectorFst<TropicalWeight>,
}

#[pymethods]
impl WeightedFst {
    #[new]
    pub fn new() -> Self {
        WeightedFst {
            fst: VectorFst::new(),
        }
    }

    pub fn set_input_symbols(&mut self, sym_vec: Vec<String>) -> () {
        let symt = SymTab::new(sym_vec);
        self.fst.set_input_symbols(Arc::new(symt.symt))
    }

    pub fn set_output_symbols(&mut self, sym_vec: Vec<String>) -> () {
        let symt = SymTab::new(sym_vec);
        self.fst.set_output_symbols(Arc::new(symt.symt))
    }

    pub fn add_states(&mut self, n: usize) {
        self.fst.add_states(n)
    }

    pub fn add_state(&mut self) -> StateId {
        self.fst.add_state()
    }

    pub fn set_start(&mut self, state: StateId) -> PyResult<()> {
        self.fst
            .set_start(state)
            .unwrap_or_else(|e| panic!("Cannot set {:?} as start state: {}", state, e));
        Ok(())
    }

    pub fn set_final(&mut self, state: StateId, weight: f32) -> PyResult<()> {
        self.fst
            .set_final(state, weight)
            .unwrap_or_else(|e| panic!("Cannot set {:?} as final state: {}", state, e));
        Ok(())
    }

    pub fn write_text(&self, path: &str) -> PyResult<()> {
        let path_output = Path::new(path);
        self.fst.write_text(path_output).unwrap();
        Ok(())
    }

    pub fn text(&self) -> String {
        match self.fst.text() {
            Ok(s) => s,
            Err(s) => panic!("Cannot produce text representation: {}", s),
        }
    }

    pub fn draw(&self, path: &str) -> PyResult<()> {
        let config = DrawingConfig::default();
        let path_output = Path::new(path);
        self.fst
            .draw(path_output, &config)
            .unwrap_or_else(|e| panic!("Cannot write to path {}: {}", path, e));
        Ok(())
    }

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

    pub fn compose(&self, other: &WeightedFst) -> PyResult<WeightedFst> {
        Ok(WeightedFst {
            fst: compose::compose(self.fst.clone(), other.fst.clone()).expect("Couldn't compose!"),
        })
    }

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

    pub fn tr_ilabel_sort(&mut self) {
        let _comp = ILabelCompare {};
        rs_tr_sort(&mut self.fst, _comp)
    }

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

#[pymodule]
#[pyo3(name = "wfst4str")]
fn wfst4str(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SymTab>()?;
    m.add_class::<WeightedFst>()?;
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
        let mut t = crate::WeightedFst::new();
        t.set_input_symbols(vec!["a", "b", "c"].iter().map(|s| s.to_string()).collect());
        t.set_output_symbols(vec!["a", "b", "c"].iter().map(|s| s.to_string()).collect());
        t.add_states(3);
        t.set_start(0).unwrap_or(());
        t.set_final(2, TropicalWeight::one()).unwrap_or(());
        let s = t.to_linear_fst("ab");
        println!("{}", s.unwrap().text())
    }
}
