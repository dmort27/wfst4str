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
use rustfst::prelude::*;

/// A representation of a wFST transition in the AT&T serialization.
#[derive(Debug, PartialEq)]
pub struct AttTransition {
    pub sourcestate: StateId,
    pub nextstate: StateId,
    pub isymbol: String,
    pub osymbol: String,
    pub weight: f32,
}

/// A representation of a wFST final state declaration in the AT&T serialization.
#[derive(Debug, PartialEq)]
pub struct AttFinalState {
    pub state: StateId,
    pub finalweight: f32,
}

/// A representation of an expression in the AT&T serialization.
#[derive(Debug, PartialEq)]
pub enum AttExpr {
    AttTr(AttTransition),
    AttFinal(AttFinalState),
    AttNone,
}

/// A parser for final state expressions in AT&T serialization.
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

/// A parser for transitions in the AT&T serialization.
pub fn att_transition(input: &str) -> IResult<&str, AttExpr> {
    let mut parser = tuple((
        recognize(digit1),
        space1,
        recognize(digit1),
        space1,
        many1(none_of(" \t\r\n")),
        space1,
        many1(none_of(" \t\r\n")),
        space1,
        float,
    ));
    let (input, (src, _, nxt, _, isym, _, osym, _, weight)) = parser(input)?;
    Ok((
        input,
        AttExpr::AttTr(AttTransition {
            sourcestate: src.parse().unwrap(),
            nextstate: nxt.parse().unwrap(),
            isymbol: isym.into_iter().collect(),
            osymbol: osym.into_iter().collect(),
            weight,
        }),
    ))
}

/// A parser for empty expressions in the AT&T serialization.
pub fn att_none(input: &str) -> IResult<&str, AttExpr> {
    let (input, _) = success("")(input)?;
    Ok((input, AttExpr::AttNone))
}

/// A parser for rows in the AT&T serialization.
pub fn att_row(input: &str) -> IResult<&str, AttExpr> {
    let mut parser = alt((att_transition, att_final_state, att_none));
    let (input, row) = parser(input)?;
    Ok((input, row))
}

/// A parser for strings/text files in the AT&T serialization.
pub fn att_file(input: &str) -> IResult<&str, Vec<AttExpr>> {
    let mut parser = separated_list0(line_ending, att_row);
    let (input, rows) = parser(input)?;
    Ok((input, rows))
}

/// Returns the number of states in the AT&T serialization of an wFST.
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
