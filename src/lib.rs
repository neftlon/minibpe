use pyo3::prelude::*;
use basic::BasicTokenizer;

mod basic;

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

#[pymodule]
#[pyo3(name = "_native")]
fn minibpe_exercise(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeBasicTokenizer>()?;
    Ok(())
}
