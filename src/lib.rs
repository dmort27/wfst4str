use nom::{
    branch::alt,
    character::complete::{digit1, line_ending, none_of, space1},
    combinator::{recognize, success},
    // error::ErrorKind,
    multi::{many1, separated_list0},
    number::complete::float,
    sequence::tuple,
    IResult,
};
use pyo3::prelude::*;
use rustfst::algorithms::compose;
use rustfst::algorithms::tr_sort as rs_tr_sort;
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::CoreFst;
use rustfst::fst_traits::SerializableFst;
use rustfst::prelude::*;
use rustfst::utils::acceptor;
use std::path::Path;
use std::sync::Arc;
// use nom::{Err, error::ErrorKind};

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
/// # Examples
///
/// ## Rust
/// ```
/// use wfst4str::WeightedFst;
///
/// let t = WeightedFst::new();
/// let sym = vec!["a", "b", "c"];
/// t.set_input_symbols(sym);
/// t.set_output_symbols(sym);
/// let q0 = t.add_state();
/// let q1 = t.add_state();
/// t.set_start(q0).unwrap();
/// t.set_final(q1, 0.0).unwrap();
/// t.add_tr(0, 1, "a", "b").unwrap();
/// ```
/// ## Python
///
/// ```python
/// import wfst4str
///
/// t = wfst4str.WeightedFst();
/// sym = ['a', 'b', 'c']
/// t.set_input_symbols(sym)
/// t.set_output_symbols(sym)
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

    pub fn get_input_symbols(&self) -> PyResult<Vec<String>> {
        Ok(self
            .fst
            .input_symbols()
            .unwrap()
            .iter()
            .map(|(_, s)| s.to_string())
            .collect())
    }

    pub fn get_output_symbols(&self) -> PyResult<Vec<String>> {
        Ok(self
            .fst
            .output_symbols()
            .unwrap()
            .iter()
            .map(|(_, s)| s.to_string())
            .collect())
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

    /// Returns the number of states in the wFST.
    pub fn num_states(&self) -> usize {
        self.fst.num_states()
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
        let symt = self.fst.input_symbols().unwrap();
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
            &syms
                .iter()
                .map(|x| symt.get_label(x).unwrap())
                .collect::<Vec<usize>>()[..],
            semiring_one(),
        );
        Ok(WeightedFst { fst: lfst })
    }

    /// Applies the wFST to a string (consisting of symbols in the wFSTs `SymbolTable`s).
    pub fn apply(&self, s: &str) -> PyResult<Vec<String>> {
        let lfst = self.to_linear_fst(s).unwrap();
        let mut fst2 = lfst.compose(self).unwrap();
        fst2.fst.set_symts_from_fst(&self.fst);
        match fst2.num_states() {
            0 => Ok(Vec::new()),
            _ => fst2.paths_as_strings(),
        }
    }

    /// Returns strings based upon the output symbols of each path
    pub fn paths_as_strings(&self) -> PyResult<Vec<String>> {
        if self.is_cyclic().unwrap() {
            panic!("wFST is cyclic. The set of all paths through it is infinite. Check your wFST for logic errors.`")
        }
        Ok(self
            .fst
            .paths_iter()
            .map(|p| {
                p.olabels
                    .iter()
                    .map(|&l| {
                        self.fst
                            .output_symbols()
                            .unwrap_or_else(|| panic!("Cannot access output SymbolTable."))
                            .get_symbol(l)
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect())
    }

    /// Returns true if the wFST has a cycle. Otherwise, it returns false.__rust_force_expr!
    pub fn is_cyclic(&self) -> PyResult<bool> {
        let fst2 = self.fst.clone();
        let mut stack: Vec<usize> = Vec::new();
        match fst2.start() {
            Some(s) => stack.push(s),
            _ => panic!("wFST lacks a start state. Aborting."),
        }
        let mut visited = vec![false; self.fst.num_states()];
        while stack.len() > 0 {
            let s = stack.pop().unwrap();
            for tr in fst2.get_trs(s).unwrap().iter() {
                if visited[tr.nextstate] {
                    return Ok(true);
                } else {
                    stack.push(tr.nextstate);
                    visited[s] = true;
                }
            }
        }
        Ok(false)
    }

    /// Replaces transitions labeled with <oth> with transitions with all unused
    /// input labels as input and output labels.
    pub fn explode_oth(&mut self) -> PyResult<()> {
        let fst2 = &mut self.fst;
        let symt = fst2.input_symbols().unwrap();
        let oth_lab = symt
            .get_label("<oth>")
            .unwrap_or_else(|| panic!("SymbolTable does not include '<oth>'"));
        let eps_lab = symt
            .get_label("<eps>")
            .unwrap_or_else(|| panic!("SymbolTable does not include '<eps>'"));
        let norm_labs: Vec<Label> = symt
            .iter()
            .map(|(x, _)| x)
            .filter(|&x| x != oth_lab && x != eps_lab)
            .collect();
        for s in fst2.states_iter() {
            let trs: Vec<Tr<TropicalWeight>> = fst2.pop_trs(s).unwrap_or_default().clone();
            let (special, normal): (Vec<Tr<TropicalWeight>>, Vec<Tr<TropicalWeight>>) = trs
                .into_iter()
                .partition(|x| x.ilabel == oth_lab || x.ilabel == eps_lab);
            let out_labs: Vec<Label> = normal.iter().map(|tr| tr.ilabel).collect();
            let complement: Vec<usize> = norm_labs
                .iter()
                .map(|&x| x)
                .filter(|x| !out_labs.contains(x))
                .collect();
            for tr in normal.iter() {
                fst2.emplace_tr(s, tr.ilabel, tr.olabel, tr.weight, tr.nextstate)
                    .unwrap();
            }
            for tr in special {
                if tr.ilabel == eps_lab {
                    fst2.emplace_tr(s, tr.ilabel, tr.olabel, tr.weight, tr.nextstate)
                        .unwrap();
                } else {
                    for lab in complement.iter() {
                        fst2.emplace_tr(s, *lab, *lab, tr.weight, tr.nextstate)
                            .unwrap();
                    }
                }
            }
        }
        Ok(())
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

    /// Returns the shortest path through the wFST.
    pub fn shortest_path(&self) -> PyResult<Vec<String>> {
        let mut shortest = WeightedFst {
            fst: shortest_path(&self.fst).unwrap(),
        };
        shortest
            .fst
            .set_input_symbols(self.fst.input_symbols().unwrap().clone());
        shortest
            .fst
            .set_output_symbols(self.fst.output_symbols().unwrap().clone());
        shortest.fst.set_symts_from_fst(&self.fst);
        shortest.paths_as_strings()
    }

    /// Populates a [`WeightedFST`] based on an AT&T description
    pub fn populate_from_att(&mut self, text: &str) -> PyResult<()> {
        if let Ok((_, exprs)) = att_file(text) {
            for expr in exprs {
                match expr {
                    AttExpr::AttTr(tr_expr) => {
                        let isymt = self
                            .fst
                            .input_symbols()
                            .unwrap_or_else(|| panic!("No input symbol table!"));
                        let osymt = self
                            .fst
                            .output_symbols()
                            .unwrap_or_else(|| panic!("No output symbol table!"));
                        let ilabel = isymt
                            .get_label(tr_expr.isymbol.clone())
                            .unwrap_or_else(|| panic!("Unkown symbol '{:?}'", tr_expr.isymbol));
                        let olabel = osymt
                            .get_label(tr_expr.osymbol.clone())
                            .unwrap_or_else(|| panic!("Unkown symbol '{:?}'", tr_expr.osymbol));
                        let tr = Tr::<TropicalWeight>::new(
                            ilabel,
                            olabel,
                            tr_expr.weight,
                            tr_expr.nextstate,
                        );
                        self.fst.add_tr(tr_expr.sourcestate, tr).unwrap_or({
                            println!(
                                "Could not create transition from {:?} to {:?}.",
                                tr_expr.sourcestate, tr_expr.nextstate
                            );
                            ()
                        });
                    }
                    AttExpr::AttFinal(fs_expr) => {
                        self.fst
                            .set_final(fs_expr.state, fs_expr.finalweight)
                            .unwrap_or_else(|e| {
                                panic!("No such state: {:?} {:?}", fs_expr.state, e)
                            });
                    }
                    AttExpr::AttNone => (),
                }
            }
        }
        Ok(())
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
pub fn wfst_from_text_string(fst_string: &str) -> PyResult<WeightedFst> {
    Ok(WeightedFst {
        fst: VectorFst::from_text_string(&fst_string).unwrap(),
    })
}

#[pyfunction]
pub fn wfst_from_text_file(path_text_fst: &str) -> PyResult<WeightedFst> {
    let fst_path = Path::new(path_text_fst);
    Ok(WeightedFst {
        fst: VectorFst::read_text(fst_path).unwrap(),
    })
}

#[derive(Debug, PartialEq)]
pub struct AttTransition {
    sourcestate: StateId,
    nextstate: StateId,
    isymbol: String,
    osymbol: String,
    weight: f32,
}

#[derive(Debug, PartialEq)]
pub struct AttFinalState {
    state: StateId,
    finalweight: f32,
}

#[derive(Debug, PartialEq)]
pub enum AttExpr {
    AttTr(AttTransition),
    AttFinal(AttFinalState),
    AttNone,
}

pub fn att_final_state(input: &str) -> IResult<&str, AttExpr> {
    let mut parser = tuple((recognize(digit1), space1, float));
    let (input, (s, _, w)) = parser(input)?;
    Ok((
        input,
        AttExpr::AttFinal(AttFinalState {
            state: s.parse().unwrap_or(0),
            finalweight: w,
        }),
    ))
}

pub fn att_transition(input: &str) -> IResult<&str, AttExpr> {
    let mut parser = tuple((
        recognize(digit1),
        space1,
        recognize(digit1),
        space1,
        many1(none_of(" \t")),
        space1,
        many1(none_of(" \t")),
        space1,
        float,
    ));
    let (input, (s, _, n, _, i, _, o, _, w)) = parser(input)?;
    Ok((
        input,
        AttExpr::AttTr(AttTransition {
            sourcestate: s.parse().unwrap(),
            nextstate: n.parse().unwrap(),
            isymbol: i.into_iter().collect(),
            osymbol: o.into_iter().collect(),
            weight: w,
        }),
    ))
}

pub fn att_none(input: &str) -> IResult<&str, AttExpr> {
    let (input, _) = success("")(input)?;
    Ok((input, AttExpr::AttNone))
}

pub fn att_row(input: &str) -> IResult<&str, AttExpr> {
    let mut parser = alt((att_transition, att_final_state, att_none));
    let (input, row) = parser(input)?;
    Ok((input, row))
}

pub fn att_file(input: &str) -> IResult<&str, Vec<AttExpr>> {
    let mut parser = separated_list0(line_ending, att_row);
    let (input, rows) = parser(input)?;
    Ok((input, rows))
}

pub fn att_num_states(text: &str) -> usize {
    match att_file(text) {
        Ok((_, rows)) => {
            let mut max_state = 0;
            for row in rows {
                match row {
                    AttExpr::AttTr(tr) => {
                        max_state = max_state.max(tr.sourcestate);
                        max_state = max_state.max(tr.nextstate);
                    }
                    AttExpr::AttFinal(f) => {
                        max_state = max_state.max(f.state);
                    }
                    AttExpr::AttNone => (),
                }
            }
            max_state
        }
        _ => panic!("Cannot parse string as AT&T wFST."),
    }
}

#[pymodule]
#[pyo3(name = "wfst4str")]
fn wfst4str(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SymTab>()?;
    m.add_class::<WeightedFst>()?;
    m.add_function(wrap_pyfunction!(wfst_from_text_string, m)?)?;
    m.add_function(wrap_pyfunction!(wfst_from_text_string, m)?)?;
    // m.add_function(wrap_pyfunction!(compose, m)?)?;
    // m.add_function(wrap_pyfunction!(tr_ilabel_sort, m)?)?;
    // m.add_function(wrap_pyfunction!(tr_olabel_sort, m)?)?;
    Ok(())
}
