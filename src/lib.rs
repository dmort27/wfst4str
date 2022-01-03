use pyo3::prelude::*;
use rustfst::algorithms::determinize::{
    determinize_with_config, DeterminizeConfig, DeterminizeType,
};
use rustfst::algorithms::tr_sort as rs_tr_sort;
use rustfst::algorithms::{closure, compose, concat, invert, project, union, ProjectType, rm_epsilon};
use rustfst::algorithms::{minimize_with_config, MinimizeConfig};
use rustfst::fst_impls::VectorFst;
use rustfst::fst_properties::FstProperties;
use rustfst::fst_traits::{CoreFst, ExpandedFst, SerializableFst};
use rustfst::prelude::*;
use rustfst::semirings::TropicalWeight;
use rustfst::utils::{acceptor, transducer};
use rustfst::KSHORTESTDELTA;
use std::collections::HashSet;
// use std::iter::FromIterator;
use rand::Rng;
use std::iter::FromIterator;
use std::path::Path;
use std::sync::Arc;

pub mod att_parse;

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
        SymTab { symt }
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

impl Default for WeightedFst {
    fn default() -> Self {
        WeightedFst::new()
    }
}

#[pymethods]
impl WeightedFst {
    /// Constructs a [`WeightedFst`] object.
    ///
    /// # Examples
    ///
    /// ## Python
    ///
    /// ```python
    /// import wfst4str
    /// t = wfst4str.WeightedFst()
    /// sym = ['a', 'b', 'c']
    /// t.set_input_symbols(sym)
    /// t.set_output_symbols(sym)
    /// q0 = t.add_state()
    /// q1 = t.add_state()
    /// t.set_start(q0)
    /// q1.set_final(q1, 0.0)
    /// t.add_tr(0, 1, 'a', 'b')
    /// assert (t.apply('a') == 'b')
    /// ```
    #[new]
    pub fn new() -> Self {
        WeightedFst {
            fst: VectorFst::new(),
        }
    }

    /// Creates a symbol table from a vector of strings and associates it with the wFST.
    pub fn set_input_symbols(&mut self, sym_vec: Vec<String>) {
        let symt = SymTab::new(sym_vec);
        self.fst.set_input_symbols(Arc::new(symt.symt))
    }

    /// Creates a symbol table from a vector of strings and associates it with the wFST.
    pub fn set_output_symbols(&mut self, sym_vec: Vec<String>) {
        let symt = SymTab::new(sym_vec);
        self.fst.set_output_symbols(Arc::new(symt.symt))
    }

    /// Returns the input symbols of the wFST as a list
    pub fn get_input_symbols(&self) -> PyResult<Vec<String>> {
        Ok(self
            .fst
            .input_symbols()
            .unwrap_or(&Arc::new(SymbolTable::new()))
            .iter()
            .map(|(_, s)| s.to_string())
            .collect())
    }

    /// Returns the output symbols of the wFST as a list
    pub fn get_output_symbols(&self) -> PyResult<Vec<String>> {
        Ok(self
            .fst
            .output_symbols()
            .unwrap_or(&Arc::new(SymbolTable::new()))
            .iter()
            .map(|(_, s)| s.to_string())
            .collect())
    }

    /// Adds `n` states to the wFST.
    pub fn add_states(&mut self, n: usize) {
        self.fst.add_states(n)
    }

    /// Adds one new state and returns the corresponding ID (an integer).
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
    pub fn num_states(&self) -> StateId {
        self.fst.num_states()
    }

    /// Serializes the wFST to a text file at `path` in AT&T format (OpenFST compatible).
    pub fn write_text(&self, path: &str) -> PyResult<()> {
        let path_output = Path::new(path);
        self.fst
            .write_text(path_output)
            .unwrap_or_else(|e| panic!("Could not write to {}: {}", path, e));
        Ok(())
    }

    /// Returns a serialization of the wFST as a string in AT&T format (OpenFST compatible).
    pub fn text(&self) -> String {
        match self.fst.text() {
            Ok(s) => s,
            Err(e) => panic!("Cannot produce text representation: {}", e),
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

    /// Adds a small amount of random noise to the weight of each transition.
    pub fn add_noise(&mut self) -> PyResult<()> {
        let mut rng = rand::thread_rng();
        for s in self.fst.states_iter() {
            let trs: Vec<Tr<TropicalWeight>> = self.fst.pop_trs(s).unwrap_or_default().clone();
            for tr in trs.iter() {
                let noise = TropicalWeight::new(rng.gen::<f32>() * 0.0001);
                let new_weight = tr.weight.times(noise).unwrap_or(tr.weight);
                self.fst
                    .emplace_tr(s, tr.ilabel, tr.olabel, new_weight, tr.nextstate)
                    .unwrap_or_else(|e| panic!("Cannot create Tr: {}", e));
            }
        }
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
        self.fst.add_tr(source_state, tr).unwrap_or_else(|e| {
            println!(
                "Cannot add Tr from {:?} to {:?}: {}",
                source_state, next_state, e
            )
        });
        Ok(())
    }

    // Algorithms

    /// Kleene closure of a wFST via mutation
    pub fn closure_in_place_star(&mut self) {
        closure::closure(&mut self.fst, closure::ClosureType::ClosureStar)
    }

    /// Kleene plus closure of a wFST via mutation
    pub fn closure_in_place_plus(&mut self) {
        closure::closure(&mut self.fst, closure::ClosureType::ClosurePlus)
    }

    /// Returns the Kleene closure of a wFST
    pub fn closure_star(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        closure::closure(&mut fst, closure::ClosureType::ClosureStar);
        Ok(WeightedFst { fst })
    }

    /// Returns the Kleene plus closure of a wFST
    pub fn closure_plus(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        closure::closure(&mut fst, closure::ClosureType::ClosurePlus);
        Ok(WeightedFst { fst })
    }

    /// Returns the composition of the wFST and another wFST (`other`)
    pub fn compose(&mut self, other: &mut WeightedFst) -> PyResult<WeightedFst> {
        self.tr_olabel_sort();
        other.tr_ilabel_sort();
        Ok(WeightedFst {
            fst: compose::compose(self.fst.clone(), other.fst.clone()).expect("Couldn't compose!"),
        })
    }

    /// Returns the concatentaion of the wFST and another wFST (`other`)
    pub fn concat(&self, other: &WeightedFst) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        concat::concat(&mut fst, &other.fst).expect("Cannot concatenate wFSTs!");
        Ok(WeightedFst { fst })
    }

    /// Returns a determinized wFST weakly equivalent to `self`.
    pub fn determinize(&self) -> PyResult<WeightedFst> {
        let mut fst: VectorFst<TropicalWeight> = determinize_with_config(
            &self.fst,
            DeterminizeConfig::new(0.001, DeterminizeType::DeterminizeDisambiguate),
        )
        .expect("Could not determinize wFST");
        fst.set_properties(
            fst.properties() | FstProperties::I_DETERMINISTIC | FstProperties::O_DETERMINISTIC,
        );
        Ok(WeightedFst { fst })
    }

    /// Concatenates a wFST (`other`) to the wFST in place.
    pub fn concat_in_place(&mut self, other: &WeightedFst) {
        concat::concat(&mut self.fst, &other.fst).expect("Cannot concatenate wFST!")
    }

    /// Returns the inversion of a wFST.
    pub fn invert(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        invert(&mut fst);
        Ok(WeightedFst { fst })
    }

    /// Inverts a wFST in place.
    pub fn invert_in_place(&mut self) {
        invert(&mut self.fst)
    }

    /// Returns a minimized wFST. Minimizes any deterministic wFST. Also
    /// minimizes non-deterministic wFSTs if they use an idempotent semiring.
    pub fn minimize(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        minimize_with_config(&mut fst, MinimizeConfig::new(KSHORTESTDELTA, true))
            .expect("Cannot minimize wFST!");
        Ok(WeightedFst { fst })
    }

    /// Minimizes a deterministic wFST in place. Also minizes non-deterministic
    /// wFSTs if they use an idempotent semiring.
    pub fn minimize_in_place(&mut self) -> PyResult<()> {
        minimize_with_config(&mut self.fst, MinimizeConfig::new(KSHORTESTDELTA, true))
            .expect("Cannot minimize wFST!");
        Ok(())
    }

    /// Project the input labels of a wFST, replacing the output labels with them.
    pub fn project_input(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        project(&mut fst, ProjectType::ProjectInput);
        Ok(WeightedFst { fst })
    }

    /// Project the input labels of a wFST, replacing the output labels with them.
    pub fn project_output(&self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        project(&mut fst, ProjectType::ProjectOutput);
        Ok(WeightedFst { fst })
    }

    /// In-place input projection of the wFST.
    pub fn project_in_place_input(&mut self) {
        project(&mut self.fst, ProjectType::ProjectInput);
    }

    /// In-place output projection of the wFST.
    pub fn project_in_place_output(&mut self) {
        project(&mut self.fst, ProjectType::ProjectOutput);
    }

    /// Returns the union of the wFST and another (`other`).
    pub fn union(&self, other: &WeightedFst) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        union::union(&mut fst, &other.fst).expect("Cannot union wFSTs!");
        Ok(WeightedFst { fst })
    }

    /// In-place union of the wFST and another (`other`).
    pub fn union_in_place(&mut self, other: &WeightedFst) {
        union::union(&mut self.fst, &other.fst).expect("Cannot union wFSTs!");
    }

    /// Returns a wFST with epsilon-transitions (transitions with epsilon as
    /// both input and output labels) removed.
    pub fn rm_epsilon(&mut self) -> PyResult<WeightedFst> {
        let mut fst = self.fst.clone();
        rm_epsilon::rm_epsilon(&mut fst).expect("Cannot remove epsilons!");
        Ok(WeightedFst { fst })
    }

    /// Removes epsilon transitions in place.
    pub fn rm_epsilon_in_place(&mut self) -> PyResult<()> {
        rm_epsilon::rm_epsilon(&mut self.fst).expect("Cannot remove epsilons!");
        Ok(())
    }

    /// Converts a string to a linear wFST using the input `SymbolTable` of the wFST.
    // pub fn to_linear_fst_old(&self, s: &str) -> PyResult<WeightedFst> {
    //     let symt = self
    //         .fst
    //         .input_symbols()
    //         .expect("wFST lacks input symbol table.");
    //     let mut syms: Vec<String> = Vec::new();
    //     let mut acc: String = String::from("");
    //     for c in s.chars() {
    //         acc.push(c);
    //         if !symt.contains_symbol(&acc) {
    //             if let Some(n) = acc.pop() {
    //                 syms.push(acc);
    //                 acc = n.to_string();
    //             }
    //         }
    //     }
    //     syms.push(acc);
    //     fn semiring_one<W: Semiring>() -> W {
    //         W::one()
    //     }
    //     let lfst: VectorFst<TropicalWeight> = acceptor(
    //         &syms
    //             .iter()
    //             .map(|x| {
    //                 symt.get_label(x)
    //                     .unwrap_or_else(|| panic!("Input symbol table lacks symbol \"{}\".", x))
    //             })
    //             .collect::<Vec<Label>>()[..],
    //         semiring_one(),
    //     );
    //     Ok(WeightedFst { fst: lfst })
    // }

    // pub fn to_linear_fst2(&self, s: &str) -> PyResult<WeightedFst> {
    //     fn semiring_one<W: Semiring>() -> W {
    //         W::one()
    //     }
    //     let symt = self
    //         .fst
    //         .input_symbols()
    //         .expect("wFST lacks input symbol table.");
    //     let syms: Vec<String> = s.chars().map(|x| x.to_string()).collect();
    //     let fst: VectorFst<TropicalWeight> = acceptor(
    //         &syms
    //             .iter()
    //             .map(|x| {
    //                 symt.get_label(x)
    //                     .unwrap_or_else(|| panic!("Input symbol table lacks symbol \"{}\".", x))
    //             })
    //             .collect::<Vec<Label>>(),
    //         semiring_one(),
    //     );
    //     Ok(WeightedFst { fst })
    // }

    /// Takes an string and returns a corresponding linear wFST (an acceptor,
    /// since the input labels are the same as the output labels and it,
    /// therefore, is functionally a wFSA).
    pub fn to_linear_acceptor(&self, s: &str) -> PyResult<WeightedFst> {
        let labs = self.isyms_to_labs(s);
        let fst: VectorFst<TropicalWeight> = acceptor(&labs, TropicalWeight::one());
        Ok(WeightedFst { fst })
    }

    /// Returns a wFST that transduces between `s1` and `s2`.
    pub fn to_linear_transducer(&self, s1: &str, s2: &str) -> PyResult<WeightedFst> {
        let labs1 = self.isyms_to_labs(s1);
        let labs2 = self.isyms_to_labs(s2);
        let max_len = labs1.len().max(labs2.len());
        let labs1: Vec<Label> = labs1
            .into_iter()
            .chain(std::iter::repeat(0))
            .take(max_len)
            .collect();
        let labs2: Vec<Label> = labs2
            .into_iter()
            .chain(std::iter::repeat(0))
            .take(max_len)
            .collect();
        let fst: VectorFst<TropicalWeight> =
            transducer(&labs1[..], &labs2[..], TropicalWeight::one());
        Ok(WeightedFst { fst })
    }

    pub fn isyms_to_labs(&self, s: &str) -> Vec<Label> {
        let symt = self
            .fst
            .input_symbols()
            .expect("wFST lacks input symbol table.");
        s.chars()
            .map(|x| {
                symt.get_label(x.to_string())
                    .unwrap_or_else(|| panic!("Input symbol table lacks symbol \"{}\".", x))
            })
            .collect::<Vec<Label>>()
    }

    /// Applies the wFST to a string (consisting of symbols in the wFSTs `SymbolTable`s).
    pub fn apply(&mut self, s: &str) -> PyResult<HashSet<String>> {
        let mut lfst = self
            .to_linear_acceptor(s)
            .unwrap_or_else(|e| panic!("Cannot linearize \"{}\": {}", s, e));
        let mut fst2 = lfst.compose(self).expect("Cannot compose wFSTs.");
        fst2.fst.set_symts_from_fst(&self.fst);
        match fst2.num_states() {
            0 => Ok(HashSet::new()),
            _ => fst2.paths_as_strings(),
        }
    }

    pub fn strings_for_shortest_paths(&mut self, s: &str) -> PyResult<HashSet<String>> {
        let mut fst = self
            .to_linear_acceptor(s)
            .unwrap_or_else(|e| panic!("Cannot linearize \"{}\": {}", s, e));
        let mut fst2 = fst.compose(self).expect("Cannot compose wFSTs.");
        fst2.fst.set_symts_from_fst(&self.fst);
        match fst2.num_states() {
            0 => Ok(HashSet::new()),
            _ => fst2.shortest_path(),
        }
    }

    /// Returns strings based upon the output symbols of each path
    pub fn paths_as_strings(&self) -> PyResult<HashSet<String>> {
        if self.is_cyclic().unwrap() {
            panic!("wFST is cyclic. The set of all paths through it is infinite. Check your wFST for logic errors.`")
        }
        Ok(HashSet::from_iter(self.fst.paths_iter().map(|p| {
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
        })))
    }

    /// Returns true if the wFST has a cycle. Otherwise, it returns false.
    pub fn is_cyclic(&self) -> PyResult<bool> {
        let fst2 = self.fst.clone();
        let mut stack: Vec<StateId> = Vec::new();
        match fst2.start() {
            Some(s) => stack.push(s),
            _ => panic!("wFST lacks a start state. Aborting."),
        }
        let mut visited = vec![false; self.fst.num_states()];
        while !stack.is_empty() {
            let s = stack.pop().unwrap();
            for tr in fst2
                .get_trs(s)
                .unwrap_or_else(|e| panic!("State {} not present in wFST: {}", s, e))
                .iter()
            {
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
    fn explode_oth_old(&mut self) -> PyResult<()> {
        let fst2 = &mut self.fst;
        let empty_symt = Arc::new(SymbolTable::new());
        let symt = fst2.input_symbols().unwrap_or(&empty_symt);
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
            let complement: Vec<Label> = norm_labs
                .iter()
                .copied()
                .filter(|x| !out_labs.contains(x))
                .collect();
            for tr in normal.iter() {
                fst2.emplace_tr(s, tr.ilabel, tr.olabel, tr.weight, tr.nextstate)
                    .unwrap_or_else(|e| {
                        panic!("Cannot create Tr; state {} does not exist: {}", s, e)
                    });
            }
            for tr in special {
                if tr.ilabel == eps_lab {
                    fst2.emplace_tr(s, tr.ilabel, tr.olabel, tr.weight, tr.nextstate)
                        .unwrap_or_else(|e| {
                            panic!("Cannot create Tr; state {} does not exist: {}", s, e)
                        });
                } else {
                    for lab in complement.iter() {
                        fst2.emplace_tr(s, *lab, *lab, tr.weight, tr.nextstate)
                            .unwrap_or_else(|e| {
                                panic!("Cannot create Tr; state {} does not exist: {}", s, e)
                            });
                    }
                }
            }
        }
        Ok(())
    }

    /// Replaces transitions labeled with <oth> with transitions with all unused
    /// input labels as input and output labels.
    pub fn explode_oth(&mut self, special: HashSet<String>) -> PyResult<()> {
        let fst = &mut self.fst; // Make a mutual reference to the inner field of &self to reduce boilerplate
        let empty_symt = Arc::new(SymbolTable::new());
        let symt = fst.input_symbols().unwrap_or(&empty_symt);
        let special = &mut special.clone();
        special.insert("<eps>".to_string());
        special.insert("<oth>".to_string());
        let oth_label = symt
            .get_label("<oth>")
            .unwrap_or_else(|| panic!("SymbolTable does not include '<oth>'"));
        let normal_set: HashSet<Label> = symt
                .iter()
                .filter(|(_, s)| !special.contains(&s.to_string()))
                .map(|(x, _)| x)
                .into_iter()
                .collect();
        for s in fst.states_iter() {
            let trs: Vec<Tr<TropicalWeight>> = fst.pop_trs(s).unwrap_or_default().clone();
            let outbound: HashSet<Label> = trs.iter().map(|tr| tr.ilabel).collect();
            let difference: HashSet<Label> = normal_set.difference(&outbound).copied().collect();
            for tr in trs.iter() {
                if tr.ilabel == oth_label {
                    for &lab in &difference {
                        fst.emplace_tr(s, lab, lab, tr.weight, tr.nextstate)
                            .unwrap_or_else(|e| panic!("Cannot create Tr: {}", e));
                    }
                } else {
                    fst.emplace_tr(s, tr.ilabel, tr.olabel, tr.weight, tr.nextstate)
                        .unwrap_or_else(|e| panic!("Cannot create Tr: {}", e));
                }
            }
        }
        Ok(())
    }

    /// Replace transitions with with input symbol `sym` (e.g., "<v>") with a
    /// set of transitions in which the the input symbols consist of the symbols
    /// in `syms` (e.g., vec!["a", "e", "i", "o", "u"]).
    pub fn sub(&mut self, sym: String, syms: Vec<String>) {
        let fst = &mut self.fst;
        let empty_symt = Arc::new(SymbolTable::new());
        let symt = fst.input_symbols().unwrap_or(&empty_symt);
        let lab = symt
            .get_label(&sym)
            .unwrap_or_else(|| panic!("Symbol table does not include \"{}\"!", sym));
        let labs: Vec<Label> = syms
            .iter()
            .map(|x| {
                symt.get_label(&x)
                    .unwrap_or_else(|| panic!("Symbol table does not include \"{}\"!", x))
            })
            .collect();
        for s in fst.states_iter() {
            let trs: Vec<Tr<TropicalWeight>> = fst.pop_trs(s).unwrap_or_default().clone();
            for tr in trs {
                if tr.ilabel == lab {
                    for &l in labs.iter() {
                        fst.emplace_tr(s, l, l, tr.weight, tr.nextstate)
                            .unwrap_or_else(|e| {
                                panic!("Cannot emplace Tr from state {}: {}", s, e)
                            });
                    }
                } else {
                    fst.add_tr(s, tr)
                        .unwrap_or_else(|e| panic!("Cannot add Tr from {}: {}", s, e));
                }
            }
        }
    }

    /// Sorts the transitions of a wFST based on its input labels.
    pub fn tr_ilabel_sort(&mut self) {
        let _comp = ILabelCompare {};
        rs_tr_sort(&mut self.fst, _comp);
        self.fst
            .set_properties(self.fst.properties() | FstProperties::I_LABEL_SORTED)
    }

    /// Sorts the transitions of a wFST based on its output labels.
    pub fn tr_olabel_sort(&mut self) {
        let _comp = OLabelCompare {};
        rs_tr_sort(&mut self.fst, _comp);
        self.fst
            .set_properties(self.fst.properties() | FstProperties::O_LABEL_SORTED)
        // self.fst.set_properties_with_mask(self.fst.properties(), FstProperties::O_LABEL_SORTED)
    }

    /// Returns the shortest path through the wFST.
    pub fn shortest_path(&self) -> PyResult<HashSet<String>> {
        let mut shortest = WeightedFst {
            fst: shortest_path(&self.fst)
                .unwrap_or_else(|e| panic!("Cannot convert wFST to shortest path: {}", e)),
        };
        shortest.fst.set_input_symbols(
            self.fst
                .input_symbols()
                .unwrap_or(&Arc::new(SymbolTable::new()))
                .clone(),
        );
        shortest.fst.set_output_symbols(
            self.fst
                .output_symbols()
                .unwrap_or(&Arc::new(SymbolTable::new()))
                .clone(),
        );
        shortest.fst.set_symts_from_fst(&self.fst);
        shortest.paths_as_strings()
    }

    /// Populates a [`WeightedFst`] based on an AT&T description
    pub fn populate_from_att(&mut self, text: &str) -> PyResult<()> {
        if let Ok((_, exprs)) = att_parse::att_file(text) {
            for expr in exprs {
                match expr {
                    att_parse::AttExpr::AttTr(tr_expr) => {
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
                            .unwrap_or_else(|| panic!("Unkown symbol {:?}", tr_expr.isymbol));
                        let olabel = osymt
                            .get_label(tr_expr.osymbol.clone())
                            .unwrap_or_else(|| panic!("Unkown symbol {:?}", tr_expr.osymbol));
                        let tr = Tr::<TropicalWeight>::new(
                            ilabel,
                            olabel,
                            tr_expr.weight,
                            tr_expr.nextstate,
                        );
                        while !(self.fst.states_iter().any(|x| x == tr_expr.sourcestate)
                            && self.fst.states_iter().any(|x| x == tr_expr.nextstate))
                        {
                            self.fst.add_state();
                        }
                        self.fst
                            .add_tr(tr_expr.sourcestate, tr)
                            .unwrap_or_else(|e| {
                                println!(
                                    "Could not create transition from {:?} to {:?}: {:?}.",
                                    tr_expr.sourcestate, tr_expr.nextstate, e
                                );
                            });
                    }
                    att_parse::AttExpr::AttFinal(fs_expr) => {
                        while !(self.fst.states_iter().any(|x| x == fs_expr.state)) {
                            self.fst.add_state();
                        }
                        self.fst
                            .set_final(fs_expr.state, fs_expr.finalweight)
                            .unwrap_or_else(|e| {
                                println!("No such state: {:?} {:?}", fs_expr.state, e)
                            });
                    }
                    att_parse::AttExpr::AttNone => (),
                }
                // println!("self.fst: {:?}", self.fst);
            }
        }
        Ok(())
    }
}

/// Returns an wFST corresponding to `fst_string` (deprecated).
#[pyfunction]
pub fn wfst_from_text_string(fst_string: &str) -> PyResult<WeightedFst> {
    Ok(WeightedFst {
        fst: VectorFst::from_text_string(fst_string)
            .unwrap_or_else(|e| panic!("Cannot deserialize wFST: {}", e)),
    })
}

/// Returns a wFST corresponding the one represented in the text file `path_text_fst` (deprecated).
#[pyfunction]
pub fn wfst_from_text_file(path_text_fst: &str) -> PyResult<WeightedFst> {
    let fst_path = Path::new(path_text_fst);
    Ok(WeightedFst {
        fst: VectorFst::read_text(fst_path)
            .unwrap_or_else(|e| panic!("Cannot read wFST at path {}: {}", path_text_fst, e)),
    })
}

#[pymodule]
fn wfst4str(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SymTab>()?;
    m.add_class::<WeightedFst>()?;
    m.add_function(wrap_pyfunction!(wfst_from_text_string, m)?)?;
    Ok(())
}
