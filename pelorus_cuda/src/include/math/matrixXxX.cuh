#pragma once

#include <cassert>
#include <core/common.cuh>
#include <core/memory.cuh>
#include <core/memory_cuda.cuh>
#include <vector>

namespace rm = rmagine;
namespace rmagine {
template <typename T, typename M> struct MatrixXxX {
  rm::Memory<T, M> m_data;
  int m_numRows;
  int m_numCols;

  virtual ~MatrixXxX() {}

  MatrixXxX(const MatrixXxX &other)
      : m_numRows(other.m_numRows), m_numCols(other.m_numCols) {
    m_data = rm::Memory<T, M>(m_numRows * m_numCols);
    other.m_data = m_data;
  }

  MatrixXxX(MatrixXxX &&other)
      : m_numRows(other.m_numRows), m_numCols(other.m_numCols) {
    m_data = other.m_data;
    // other.m_data.resize(0);
  }

  MatrixXxX(int numRows, int numCols) : m_numRows(numRows), m_numCols(numCols) {
    m_data = rm::Memory<T, M>(m_numRows * m_numCols);
  }

  JLIO_INLINE_FUNCTION
  int numRows() const { return m_numRows; }

  JLIO_INLINE_FUNCTION
  int numCols() const { return m_numCols; }

  JLIO_INLINE_FUNCTION
  T &operator()(int row, int col) {
#ifndef NDEBUG
    assert(row >= 0 && row < m_numRows);
    assert(col >= 0 && col < m_numCols);
#endif
    return m_data.at(col * m_numRows + row);
  }

  JLIO_INLINE_FUNCTION
  const T &operator()(int row, int col) const {
#ifndef NDEBUG
    assert(row >= 0 && row < m_numRows);
    assert(col >= 0 && col < m_numCols);
#endif
    return m_data.at(col * m_numRows + row);
  }
};

template <typename T, typename M> struct MatrixXxXView {
  rm::MemoryView<T, M> m_data;
  int m_numRows;
  int m_numCols;

  virtual ~MatrixXxXView() {}

  MatrixXxXView(const MatrixXxX<T, M> &other)
      : m_numRows(other.m_numRows), m_numCols(other.m_numCols),
        m_data(other.m_data) {}

  MatrixXxXView(MatrixXxX<T, M> &&other)
      : m_numRows(other.m_numRows), m_numCols(other.m_numCols) {
    m_data = other.m_data.slice(0, m_numRows * m_numCols);
    // other.m_data.resize(0);
  }

  MatrixXxXView(int numRows, int numCols)
      : m_numRows(numRows), m_numCols(numCols) {
    m_data = rm::Memory<T, M>(m_numRows * m_numCols);
  }

  JLIO_INLINE_FUNCTION
  int numRows() const { return m_numRows; }

  JLIO_INLINE_FUNCTION
  int numCols() const { return m_numCols; }

  JLIO_INLINE_FUNCTION
  T &operator()(int row, int col) {
#ifndef NDEBUG
    assert(row >= 0 && row < m_numRows);
    assert(col >= 0 && col < m_numCols);
#endif
    return m_data.at(col * m_numRows + row);
  }

  JLIO_INLINE_FUNCTION
  const T &operator()(int row, int col) const {
#ifndef NDEBUG
    assert(row >= 0 && row < m_numRows);
    assert(col >= 0 && col < m_numCols);
#endif
    return m_data.at(col * m_numRows + row);
  }
};

} // namespace rmagine
