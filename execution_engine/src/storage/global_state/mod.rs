//! Global state.

/// In-memory implementation of global state.
pub mod in_memory;

/// Lmdb implementation of global state.
pub mod lmdb;

/// Lmdb implementation of global state with cache.
pub mod scratch;

use std::{collections::HashMap, hash::BuildHasher};

use tracing::error;

use casper_hashing::Digest;
use casper_types::{
    bytesrepr::{self, FromBytes, ToBytes},
    Key, StoredValue,
};

use crate::{
    shared::{
        additive_map::AdditiveMap,
        newtypes::CorrelationId,
        transform::{self, Transform},
    },
    storage::{
        transaction_source::{Transaction, TransactionSource},
        trie::{merkle_proof::TrieMerkleProof, Trie},
        trie_store::{
            operations::{read, write, ReadResult, WriteResult},
            TrieStore,
        },
    },
};

/// A trait expressing the reading of state. This trait is used to abstract the underlying store.
pub trait StateReader<K, V> {
    /// An error which occurs when reading state
    type Error;

    /// Returns the state value from the corresponding key
    fn read(&self, correlation_id: CorrelationId, key: &K) -> Result<Option<V>, Self::Error>;

    /// Returns the merkle proof of the state value from the corresponding key
    fn read_with_proof(
        &self,
        correlation_id: CorrelationId,
        key: &K,
    ) -> Result<Option<TrieMerkleProof<K, V>>, Self::Error>;

    /// Returns the keys in the trie matching `prefix`.
    fn keys_with_prefix(
        &self,
        correlation_id: CorrelationId,
        prefix: &[u8],
    ) -> Result<Vec<K>, Self::Error>;
}

/// An error emitted by the execution engine on commit
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum CommitError {
    /// Root not found.
    #[error("Root not found: {0:?}")]
    RootNotFound(Digest),
    /// Root not found while attempting to read.
    #[error("Root not found while attempting to read: {0:?}")]
    ReadRootNotFound(Digest),
    /// Root not found while attempting to write.
    #[error("Root not found while writing: {0:?}")]
    WriteRootNotFound(Digest),
    /// Key not found.
    #[error("Key not found: {0}")]
    KeyNotFound(Key),
    /// Transform error.
    #[error(transparent)]
    TransformError(transform::Error),
}

/// Provides `commit` method.
pub trait CommitProvider: StateProvider {
    /// Applies changes and returns a new post state hash.
    /// block_hash is used for computing a deterministic and unique keys.
    fn commit(
        &self,
        correlation_id: CorrelationId,
        state_hash: Digest,
        effects: AdditiveMap<Key, Transform>,
    ) -> Result<Digest, Self::Error>;
}

/// A trait expressing operations over the trie.
pub trait StateProvider {
    /// Associated error type for `StateProvider`.
    type Error;

    /// Associated reader type for `StateProvider`.
    type Reader: StateReader<Key, StoredValue, Error = Self::Error>;

    /// Checkouts to the post state of a specific block.
    fn checkout(&self, state_hash: Digest) -> Result<Option<Self::Reader>, Self::Error>;

    /// Returns an empty root hash.
    fn empty_root(&self) -> Digest;

    /// Reads a `Trie` from the state if it is present
    fn get_trie(
        &self,
        correlation_id: CorrelationId,
        trie_key: &Digest,
    ) -> Result<Option<Trie<Key, StoredValue>>, Self::Error>;

    /// Insert a trie node into the trie
    fn put_trie(
        &self,
        correlation_id: CorrelationId,
        trie: &Trie<Key, StoredValue>,
    ) -> Result<Digest, Self::Error>;

    /// Finds all of the missing or corrupt keys of which are descendants of `trie_key`
    fn missing_trie_keys(
        &self,
        correlation_id: CorrelationId,
        trie_keys: Vec<Digest>,
    ) -> Result<Vec<Digest>, Self::Error>;
}

/// Private data structure used to update global state efficiently.
mod fancy_trie {
    use super::*;
    use crate::storage::{
        transaction_source::{Readable, Writable},
        trie::{Pointer as TriePointer, PointerBlock as TriePointerBlock},
    };
    use core::{convert::TryInto, mem};

    /// Trie node variants.
    pub enum FancyTrie<K, V> {
        Leaf(K, V),
        Extension {
            affix: Vec<u8>,
            pointer: Pointer<K, V>,
        },
        Node {
            branches: Box<[Option<Pointer<K, V>>; 256]>,
        },
    }

    /// Holds information on how to get to a trie node.
    pub enum Link<K, V> {
        /// Load the node via global state if necessary.
        GlobalState(Digest),
        /// Use a direct, in-memory, reference.
        Fancy(Box<FancyTrie<K, V>>),
    }

    impl<K, V> Link<K, V> {
        fn global_state_digest(&self) -> Option<Digest> {
            match self {
                Link::GlobalState(digest) => Some(digest.clone()),
                Link::Fancy(_) => None,
            }
        }
    }

    impl<K, V> Default for Link<K, V> {
        fn default() -> Self {
            Self::GlobalState([0; Digest::LENGTH].into())
        }
    }

    /// Links to a node while being loosely compatible with `TriePointer`.
    pub struct Pointer<K, V> {
        pub link: Link<K, V>,
        pub is_leaf: bool,
    }

    impl<K, V> Default for Pointer<K, V> {
        fn default() -> Self {
            Pointer {
                link: Default::default(),
                is_leaf: false,
            }
        }
    }

    /// Finds the size of the longest common prefix of two byte strings.
    fn longest_common_prefix_size(a: &[u8], b: &[u8]) -> usize {
        let n = core::cmp::min(a.len(), b.len());
        for i in 0..n {
            if a[i] != b[i] {
                return i;
            }
        }
        n
    }

    impl<K, V> Pointer<K, V> {
        /// Makes a pointer to a new leaf.
        fn new_leaf(key: K, value: V) -> Pointer<K, V> {
            Pointer {
                link: Link::Fancy(Box::new(FancyTrie::Leaf(key, value))),
                is_leaf: true,
            }
        }

        /// Makes a pointer to a new extension node.
        fn new_extension(affix: &[u8], pointer: Pointer<K, V>) -> Pointer<K, V> {
            Pointer {
                link: Link::Fancy(Box::new(FancyTrie::Extension {
                    affix: affix.to_vec(),
                    pointer,
                })),
                is_leaf: false,
            }
        }
    }

    fn new_branches<K, V>() -> Box<[Option<Pointer<K, V>>; 256]> {
        Box::new([
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
        ])
    }

    /// Loads a `Trie` from global state by `Digest` and returns a `Pointer` to its fancy version.
    fn load_from_global_state<S, T, K, V>(
        store: &S,
        tx: &T,
        digest: &Digest,
    ) -> Result<Pointer<K, V>, S::Error>
    where
        S: TrieStore<K, V>,
        T: Readable<Handle = S::Handle>,
        K: FromBytes,
        V: FromBytes,
        S::Error: From<T::Error>,
    {
        Ok(
            match store
                .get(tx, digest)?
                .expect("Hashes referred to must be in global state.")
            {
                Trie::Leaf { key, value } => Pointer::new_leaf(key, value),
                Trie::Extension { affix, pointer } => Pointer::new_extension(
                    &affix,
                    Pointer {
                        link: Link::GlobalState(*pointer.hash()),
                        is_leaf: matches!(pointer, TriePointer::LeafPointer(_)),
                    },
                ),
                Trie::Node { pointer_block } => {
                    let mut branches = new_branches();
                    for (i, p) in pointer_block.as_indexed_pointers() {
                        branches[i as usize] = Some(Pointer {
                            link: Link::GlobalState(*p.hash()),
                            is_leaf: matches!(p, TriePointer::LeafPointer(_)),
                        });
                    }
                    Pointer {
                        link: Link::Fancy(Box::new(FancyTrie::Node { branches })),
                        is_leaf: false,
                    }
                }
            },
        )
    }

    impl<K, V> Pointer<K, V> {
        /// Inserts a key-value pair into a fancy trie.
        /// A new root node may result, which is why this mutates a `Pointer`.
        //
        // Can't make it safe because.
        pub fn insert<S, T, E>(&mut self, store: &S, tx: &T, key: K, value: V) -> Result<(), E>
        where
            K: FromBytes + ToBytes + PartialEq + std::fmt::Debug,
            V: FromBytes,
            S: TrieStore<K, V>,
            T: Readable<Handle = S::Handle>,
            S::Error: From<T::Error>,
            E: From<S::Error> + From<bytesrepr::Error>,
        {
            let mut bytes: &[u8] = &key.to_bytes()?;
            let mut current: std::ptr::NonNull<Pointer<K, V>> = self.into();
            let mut to_assign: Option<Pointer<K, V>> = None;
            loop {
                if let Some(digest) = unsafe { current.as_ref() }.link.global_state_digest() {
                    *unsafe { current.as_mut() } = load_from_global_state(store, tx, &digest)?;
                }
                let Pointer { is_leaf, link } = unsafe { current.as_mut() };
                if let Link::Fancy(fancy_box) = link {
                    match &mut **fancy_box {
                        FancyTrie::Leaf(k, v) => {
                            debug_assert!(*is_leaf);
                            debug_assert!(bytes.is_empty(), "Tries must be prefix-free.");
                            debug_assert_eq!(*k, key);
                            *v = value;
                            break;
                        }
                        FancyTrie::Node { branches } => {
                            debug_assert!(!*is_leaf);
                            debug_assert!(!bytes.is_empty(), "Tries must be prefix-free.");
                            let branch = &mut branches[bytes[0] as usize];
                            match branch {
                                Some(pointer) => {
                                    current = pointer.into();
                                    bytes = &bytes[1..];
                                }
                                None => {
                                    *branch = Some(Pointer::new_leaf(key, value));
                                    break;
                                }
                            }
                        }
                        FancyTrie::Extension { affix, pointer } => {
                            debug_assert!(!*is_leaf);
                            let lcp = longest_common_prefix_size(bytes, &affix[..]);
                            if lcp == affix.len() {
                                current = pointer.into();
                                bytes = &bytes[lcp..];
                            } else {
                                debug_assert!(lcp < bytes.len(), "Tries must be prefix-free.");
                                let mut branches = new_branches();
                                branches[bytes[lcp] as usize] = Some(Pointer::new_leaf(key, value));
                                branches[affix[lcp] as usize] = Some(if affix.len() - lcp >= 2 {
                                    Pointer::new_extension(&affix[lcp + 1..], mem::take(pointer))
                                } else {
                                    mem::take(pointer)
                                });
                                let node = Pointer {
                                    link: Link::Fancy(Box::new(FancyTrie::Node { branches })),
                                    is_leaf: false,
                                };
                                to_assign = Some(if lcp == 0 {
                                    node
                                } else {
                                    Pointer::new_extension(&affix[..lcp], node)
                                });
                                break;
                            }
                        }
                    }
                } else {
                    unreachable!();
                }
            }
            if let Some(to_assign) = to_assign {
                *unsafe { current.as_mut() } = to_assign;
            }
            Ok(())
        }
    }

    enum DfsTask<K, V> {
        Open(std::ptr::NonNull<Pointer<K, V>>),
        Close(std::ptr::NonNull<Pointer<K, V>>),
    }

    fn trie_pointer_from(digest: &Digest, is_leaf: bool) -> TriePointer {
        if is_leaf {
            TriePointer::LeafPointer(digest.clone())
        } else {
            TriePointer::NodePointer(digest.clone())
        }
    }

    impl<K, V> Pointer<K, V> {
        /// Writes all fancy nodes in a fancy trie to global state as if they were global state trie nodes.
        /// In essence, update global state with the entries added to the fancy trie.
        pub fn update_global_state<S, T, E>(&mut self, store: &S, tx: &mut T) -> Result<(), E>
        where
            K: ToBytes + Clone,
            V: ToBytes + Clone,
            S: TrieStore<K, V>,
            T: Writable<Handle = S::Handle>,
            S::Error: From<T::Error>,
            E: From<bytesrepr::Error> + From<S::Error>,
        {
            let mut stack: Vec<DfsTask<K, V>> = vec![DfsTask::Open(self.into())];
            while let Some(mut task) = stack.pop() {
                match &mut task {
                    DfsTask::Open(current) => match &mut unsafe { current.as_mut() }.link {
                        Link::GlobalState(_) => (),
                        Link::Fancy(fancy_box) => match &mut **fancy_box {
                            FancyTrie::Leaf(_, _) => {
                                stack.push(DfsTask::Close(*current));
                            }
                            FancyTrie::Extension { affix: _, pointer } => {
                                stack.push(DfsTask::Close(*current));
                                stack.push(DfsTask::Open(pointer.into()));
                            }
                            FancyTrie::Node { branches } => {
                                stack.push(DfsTask::Close(*current));
                                for maybe_pointer in branches.iter_mut() {
                                    if let Some(pointer) = maybe_pointer {
                                        stack.push(DfsTask::Open(pointer.into()));
                                    }
                                }
                            }
                        },
                    },
                    DfsTask::Close(current) => {
                        let current = unsafe { current.as_mut() };
                        match mem::take(&mut current.link) {
                            Link::GlobalState(_) => unreachable!(),
                            Link::Fancy(fancy_box) => match *fancy_box {
                                FancyTrie::Leaf(key, value) => {
                                    let trie = Trie::Leaf { key, value };
                                    let trie_hash = Digest::hash(trie.to_bytes()?);
                                    store.put(tx, &trie_hash, &trie)?;
                                    current.link = Link::GlobalState(trie_hash);
                                }
                                FancyTrie::Extension { affix, pointer } => match pointer.link {
                                    Link::Fancy(_) => unreachable!(),
                                    Link::GlobalState(digest) => {
                                        let trie = Trie::Extension {
                                            affix: (&affix[..]).into(),
                                            pointer: trie_pointer_from(&digest, current.is_leaf),
                                        };
                                        let trie_hash = Digest::hash(trie.to_bytes()?);
                                        store.put(tx, &trie_hash, &trie)?;
                                        current.link = Link::GlobalState(trie_hash);
                                    }
                                },
                                FancyTrie::Node { branches } => {
                                    let trie = Trie::Node{pointer_block: Box::new(TriePointerBlock::from_indexed_pointers(
                                        &branches.iter().enumerate().filter_map(|(i, maybe_pointer)| {
                                            maybe_pointer.as_ref().map(|pointer| {
                                                match pointer.link {
                                                    Link::Fancy(_) => unreachable!(),
                                                    Link::GlobalState(digest) => (i.try_into().expect("Branch indices must fit into an `u8`."), trie_pointer_from(&digest, current.is_leaf)),
                                                }
                                            })
                                        }).collect::<Vec<_>>()[..]
                                    ))};
                                    let trie_hash = Digest::hash(trie.to_bytes()?);
                                    store.put(tx, &trie_hash, &trie)?;
                                    current.link = Link::GlobalState(trie_hash);
                                }
                            },
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

/// Write multiple key/stored value pairs to the store in a single rw transaction.
pub fn put_stored_values<'a, R, S, E>(
    environment: &'a R,
    store: &S,
    correlation_id: CorrelationId,
    prestate_hash: Digest,
    stored_values: HashMap<Key, StoredValue>,
) -> Result<Digest, E>
where
    R: TransactionSource<'a, Handle = S::Handle>,
    S: TrieStore<Key, StoredValue>,
    S::Error: From<R::Error>,
    E: From<R::Error> + From<S::Error> + From<bytesrepr::Error> + From<CommitError>,
{
    let mut txn = environment.create_read_write_txn()?;
    let state_root = prestate_hash;
    let mut fancy_trie = fancy_trie::Pointer {
        link: fancy_trie::Link::GlobalState(state_root),
        // This doesn't matter because...
        is_leaf: false,
    };
    for (key, value) in stored_values.iter() {
        fancy_trie.insert::<_, _, E>(store, &txn, *key, value.clone())?;
    }
    fancy_trie.update_global_state::<_, _, E>(store, &mut txn)?;
    txn.commit()?;
    Ok(state_root)
}

/// Commit `effects` to the store.
pub fn commit<'a, R, S, H, E>(
    environment: &'a R,
    store: &S,
    correlation_id: CorrelationId,
    prestate_hash: Digest,
    effects: AdditiveMap<Key, Transform, H>,
) -> Result<Digest, E>
where
    R: TransactionSource<'a, Handle = S::Handle>,
    S: TrieStore<Key, StoredValue>,
    S::Error: From<R::Error>,
    E: From<R::Error> + From<S::Error> + From<bytesrepr::Error> + From<CommitError>,
    H: BuildHasher,
{
    let mut txn = environment.create_read_write_txn()?;
    let mut state_root = prestate_hash;

    let maybe_root: Option<Trie<Key, StoredValue>> = store.get(&txn, &state_root)?;

    if maybe_root.is_none() {
        return Err(CommitError::RootNotFound(prestate_hash).into());
    };

    for (key, transform) in effects.into_iter() {
        let read_result = read::<_, _, _, _, E>(correlation_id, &txn, store, &state_root, &key)?;

        let value = match (read_result, transform) {
            (ReadResult::NotFound, Transform::Write(new_value)) => new_value,
            (ReadResult::NotFound, transform) => {
                error!(
                    ?state_root,
                    ?key,
                    ?transform,
                    "Key not found while attempting to apply transform"
                );
                return Err(CommitError::KeyNotFound(key).into());
            }
            (ReadResult::Found(current_value), transform) => match transform.apply(current_value) {
                Ok(updated_value) => updated_value,
                Err(err) => {
                    error!(
                        ?state_root,
                        ?key,
                        ?err,
                        "Key found, but could not apply transform"
                    );
                    return Err(CommitError::TransformError(err).into());
                }
            },
            (ReadResult::RootNotFound, transform) => {
                error!(
                    ?state_root,
                    ?key,
                    ?transform,
                    "Failed to read state root while processing transform"
                );
                return Err(CommitError::ReadRootNotFound(state_root).into());
            }
        };

        let write_result =
            write::<_, _, _, _, E>(correlation_id, &mut txn, store, &state_root, &key, &value)?;

        match write_result {
            WriteResult::Written(root_hash) => {
                state_root = root_hash;
            }
            WriteResult::AlreadyExists => (),
            WriteResult::RootNotFound => {
                error!(?state_root, ?key, ?value, "Error writing new value");
                return Err(CommitError::WriteRootNotFound(state_root).into());
            }
        }
    }

    txn.commit()?;

    Ok(state_root)
}
