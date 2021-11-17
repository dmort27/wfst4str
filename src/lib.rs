use pyo3::prelude::*;
use rustfst::prelude::*;
use rustfst::algorithms::compose::compose as rscompose;
use rustfst::algorithms::tr_sort as rs_tr_sort;
use std::sync::Arc;
use std::path::Path;

/// Wraps a [`rustfst`] SymbolTable struct as a Python class.
/// 
/// # Example
/// 
/// ```
/// let symt = SymTab::new(vec!["a", "b", "c"]);
/// assert_eq!(symt.get_symbol(1), "a");
/// ```
#[pyclass]
pub struct SymTab {
    /// A SymbolTable struct from [`rustfst`]
    symt: SymbolTable,
}
#[pymethods]
impl SymTab {

    /// Constructs a new SymTab instance
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

    /// Adds a symbol to a SymTab with an associated label
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
        self.fst.set_start(state).unwrap_or_else(|e| panic!("Cannot set {:?} as start state: {}", state, e));
        Ok(())
    }

    pub fn set_final(&mut self, state: StateId, weight: f32) -> PyResult<()> {
        self.fst.set_final(state, weight).unwrap_or_else(|e| panic!("Cannot set {:?} as final state: {}", state, e));
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
        self.fst.draw(path_output, &config).unwrap_or_else(|e| panic!("Cannot write to path {}: {}", path, e));
        Ok(())
    }

    pub fn add_tr(&mut self, isym: &str, osym: &str, source_state: StateId, next_state: StateId, weight: f32) -> PyResult<()> {
        let ilabel = &self.fst.input_symbols().unwrap().get_label(isym).unwrap();
        let olabel = &self.fst.output_symbols().unwrap().get_label(osym).unwrap();
        let tr = Tr::<TropicalWeight>::new(*ilabel, *olabel, weight, next_state);
        self.fst.add_tr(source_state, tr).unwrap();
        Ok(())
    }

}

#[pyfunction]
pub fn compose(fst1: &WeightedFst, fst2: &WeightedFst) -> WeightedFst {
    match rscompose(fst1.fst.clone(), fst2.fst.clone()) {
        Ok(fst) => WeightedFst { fst },
        Err(e) => panic!("Cannot compse WFSTs: {}", e),
    }
}

#[pyfunction]
pub fn tr_ilabel_sort(fst: &mut WeightedFst) {
    let _comp = ILabelCompare {};
    rs_tr_sort(&mut fst.fst, _comp)
}

#[pyfunction]
pub fn tr_olabel_sort(fst: &mut WeightedFst) {
    let _comp = OLabelCompare {};
    rs_tr_sort(&mut fst.fst, _comp)
}

#[pymodule]
#[pyo3(name = "wfst4str")]
fn wfst4str(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SymTab>()?;
    m.add_class::<WeightedFst>()?;
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    m.add_function(wrap_pyfunction!(tr_ilabel_sort, m)?)?;
    m.add_function(wrap_pyfunction!(tr_olabel_sort, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_get_symbol() {
        let st = crate::SymTab::new(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(st.get_symbol(1).unwrap_or(""), "a");
        assert_eq!(st.get_symbol(2).unwrap_or(""), "b");
        return ()
    }

}
