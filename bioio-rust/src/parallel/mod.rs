//! Parallel processing utilities
//!
//! This module provides utilities for parallel image processing
//! using rayon thread pools.

use rayon::prelude::*;
use crate::error::{BioIoError, Result};

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    /// Stack size per thread in bytes
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub name_prefix: Option<String>,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            stack_size: None,
            name_prefix: None,
        }
    }
}

impl ThreadPoolConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Build a rayon thread pool with this configuration
    pub fn build_pool(&self) -> Result<rayon::ThreadPool> {
        let mut builder = rayon::ThreadPoolBuilder::new();

        if self.num_threads > 0 {
            builder = builder.num_threads(self.num_threads);
        }

        if let Some(stack_size) = self.stack_size {
            builder = builder.stack_size(stack_size);
        }

        if let Some(prefix) = &self.name_prefix {
            let prefix = prefix.clone();
            builder = builder.thread_name(move |idx| format!("{}-{}", prefix, idx));
        }

        builder.build().map_err(|e| BioIoError::ThreadPool(e.to_string()))
    }
}

/// Parallel chunk processor
pub struct ChunkProcessor {
    pool: rayon::ThreadPool,
}

impl ChunkProcessor {
    /// Create a new chunk processor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ThreadPoolConfig::default())
    }

    /// Create a new chunk processor with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Result<Self> {
        let pool = config.build_pool()?;
        Ok(Self { pool })
    }

    /// Create a new chunk processor with specified thread count
    pub fn with_threads(num_threads: usize) -> Result<Self> {
        Self::with_config(ThreadPoolConfig::new().with_threads(num_threads))
    }

    /// Process chunks in parallel
    pub fn process<T, F, R>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + Sync,
        F: Fn(T) -> R + Send + Sync,
        R: Send,
    {
        self.pool.install(|| {
            items.into_par_iter().map(f).collect()
        })
    }

    /// Process chunks in parallel with error handling
    pub fn try_process<T, F, R, E>(&self, items: Vec<T>, f: F) -> std::result::Result<Vec<R>, E>
    where
        T: Send + Sync,
        F: Fn(T) -> std::result::Result<R, E> + Send + Sync,
        R: Send,
        E: Send,
    {
        self.pool.install(|| {
            items.into_par_iter().map(f).collect()
        })
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}

impl Default for ChunkProcessor {
    fn default() -> Self {
        Self::new().expect("Failed to create default chunk processor")
    }
}

/// Parallel batch reader for reading multiple files
pub fn read_batch_parallel<P, R, F>(
    paths: &[P],
    num_threads: usize,
    reader_fn: F,
) -> Result<Vec<R>>
where
    P: AsRef<std::path::Path> + Sync,
    R: Send,
    F: Fn(&std::path::Path) -> Result<R> + Send + Sync,
{
    let pool = ThreadPoolConfig::new()
        .with_threads(num_threads)
        .build_pool()?;

    pool.install(|| {
        paths
            .par_iter()
            .map(|p| reader_fn(p.as_ref()))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::new().with_threads(4);
        assert_eq!(config.num_threads, 4);

        let pool = config.build_pool();
        assert!(pool.is_ok());
    }

    #[test]
    fn test_chunk_processor() {
        let processor = ChunkProcessor::new().unwrap();

        let items: Vec<i32> = (0..100).collect();
        let results: Vec<i32> = processor.process(items, |x| x * 2);

        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0);
        assert_eq!(results[50], 100);
    }

    #[test]
    fn test_chunk_processor_with_threads() {
        let processor = ChunkProcessor::with_threads(2).unwrap();
        assert!(processor.num_threads() >= 1);
    }
}
