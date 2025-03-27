use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn micro_swap_boosted(py: Python, dag: PyObject) -> PyResult<()> {
    let args = (0,);
    let res = dag.call_method1(py, "get", args)?;
    println!("{:?}", res.call_method0(py, "__repr__")?.extract::<String>(py)?);
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn microscope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(micro_swap_boosted, m)?)?;
    Ok(())
}
