use basic::BasicTokenizer;
use regex::tokenizer_with_presplit;
use pyo3::prelude::*;

mod basic;
mod regex;
mod utils;

#[derive(Clone, Debug, PartialEq)]
enum TrainError {
    NotEnoughPairs,
}

#[pyclass(frozen)]
struct NativeBasicTokenizer(BasicTokenizer);

#[pymethods]
impl NativeBasicTokenizer {
    #[staticmethod]
    fn train(text: &str, vocab_size: u32) -> Self {
        let tok_or_err = BasicTokenizer::train(text, vocab_size);
        let tok = tok_or_err.expect("training failed"); // TODO: throw Python exception
        Self(tok)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
    }

    fn decode(&self, tokens: Vec<u32>) -> String {
        self.0.decode(tokens)
    }
}

#[pyfunction]
fn py_tokenizer_with_presplit(
    text: &str,
    vocab_size: u32,
    regex: &str,
) -> NativeBasicTokenizer {
    let tok = tokenizer_with_presplit(text, vocab_size, regex);
    let tok = tok.expect("training failed");
    NativeBasicTokenizer(tok)
}

#[pymodule]
#[pyo3(name = "_native")]
fn minibpe_exercise(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeBasicTokenizer>()?;
    m.add_function(wrap_pyfunction!(py_tokenizer_with_presplit, m)?)?;
    Ok(())
}
