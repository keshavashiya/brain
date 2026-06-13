use super::{Fact, SemanticError, SemanticStore};

impl SemanticStore {
    /// Get a fact by ID from SQLite.
    ///
    /// Returns `None` if the fact does not exist or its content cannot be decrypted.
    pub fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let result = conn.query_row(
                "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                 FROM semantic_facts WHERE id = ?1",
                [fact_id],
                |row| {
                    let raw_object: String = row.get(5)?;
                    Ok((
                        Fact {
                            id: row.get(0)?,
                            namespace: row.get(1)?,
                            category: row.get(2)?,
                            subject: row.get(3)?,
                            predicate: row.get(4)?,
                            object: String::new(),
                            confidence: row.get(6)?,
                            source_episode_id: row.get(7)?,
                            agent: row.get(8)?,
                        },
                        raw_object,
                    ))
                },
            );
            match result {
                Ok((mut fact, raw_object)) => match pool.try_decrypt_content(&raw_object) {
                    Some(obj) => {
                        fact.object = obj;
                        Ok(Some(fact))
                    }
                    None => Ok(None),
                },
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })?)
    }

    /// Get all facts by category, optionally filtered by namespace.
    pub fn get_facts_by_category(
        &self,
        category: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match namespace {
                Some(ns) => (
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE category = ?1 AND (namespace = ?2 OR namespace LIKE ?3) AND superseded_by IS NULL
                     ORDER BY updated_at DESC"
                        .to_string(),
                    vec![
                        Box::new(category.to_string()),
                        Box::new(ns.to_string()),
                        Box::new(format!("{ns}/%")),
                    ],
                ),
                None => (
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE category = ?1 AND superseded_by IS NULL
                     ORDER BY updated_at DESC"
                        .to_string(),
                    vec![Box::new(category.to_string())],
                ),
            };

            let mut stmt = conn.prepare(&sql)?;
            let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                params.iter().map(|p| p.as_ref()).collect();

            let facts = stmt
                .query_map(params_ref.as_slice(), |row| {
                    let raw_object: String = row.get(5)?;
                    Ok((
                        Fact {
                            id: row.get(0)?,
                            namespace: row.get(1)?,
                            category: row.get(2)?,
                            subject: row.get(3)?,
                            predicate: row.get(4)?,
                            object: String::new(),
                            confidence: row.get(6)?,
                            source_episode_id: row.get(7)?,
                            agent: row.get(8)?,
                        },
                        raw_object,
                    ))
                })?
                .filter_map(|r| match r {
                    Ok((mut fact, raw)) => {
                        let obj = pool.try_decrypt_content(&raw)?;
                        fact.object = obj;
                        Some(fact)
                    }
                    Err(_) => None,
                })
                .collect::<Vec<_>>();

            Ok(facts)
        })?)
    }

    /// Get all facts about a specific subject.
    pub fn get_facts_about(&self, subject: &str) -> Result<Vec<Fact>, SemanticError> {
        self.get_facts_about_in_namespace(subject, None)
    }

    /// Get facts about a specific subject, optionally filtered by namespace.
    pub fn get_facts_about_in_namespace(
        &self,
        subject: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let row_to_raw_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(Fact, String)> {
                let raw_object: String = row.get(5)?;
                Ok((
                    Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: String::new(),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(8)?,
                    },
                    raw_object,
                ))
            };

            let decrypt_filter = |r: rusqlite::Result<(Fact, String)>| -> Option<Fact> {
                let (mut fact, raw) = r.ok()?;
                fact.object = pool.try_decrypt_content(&raw)?;
                Some(fact)
            };

            let facts: Vec<Fact> = if let Some(ns) = namespace {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE subject = ?1 AND (namespace = ?2 OR namespace LIKE ?3)
                     ORDER BY confidence DESC",
                )?;
                let prefix = format!("{ns}/%");
                let rows =
                    stmt.query_map(rusqlite::params![subject, ns, &prefix], row_to_raw_fact)?;
                rows.filter_map(decrypt_filter).collect()
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE subject = ?1
                     ORDER BY confidence DESC",
                )?;
                let rows = stmt.query_map([subject], row_to_raw_fact)?;
                rows.filter_map(decrypt_filter).collect()
            };

            Ok(facts)
        })?)
    }

    /// List all active (non-superseded) facts, optionally scoped to a namespace.
    pub fn list_all(&self) -> Result<Vec<Fact>, SemanticError> {
        self.list_by_namespace(None)
    }

    /// List all active facts in a namespace. **Unbounded** — kept for
    /// callers that genuinely need the full set (memory-summary
    /// inspection). New API surface (HTTP/gRPC) should use
    /// [`list_by_namespace_paginated`] so a multi-thousand-fact store
    /// doesn't return a single mega-response.
    pub fn list_by_namespace(&self, namespace: Option<&str>) -> Result<Vec<Fact>, SemanticError> {
        self.list_by_namespace_paginated(namespace, None, 0)
    }

    /// Paginated variant of [`list_by_namespace`]. `limit = None` means
    /// "no LIMIT clause" (matches the legacy unbounded behavior);
    /// `Some(n)` appends `LIMIT n OFFSET offset`. `offset` is ignored
    /// when `limit` is `None`.
    pub fn list_by_namespace_paginated(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let row_to_raw_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(Fact, String)> {
                let raw_object: String = row.get(5)?;
                Ok((
                    Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: String::new(),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(8)?,
                    },
                    raw_object,
                ))
            };

            let decrypt_filter = |r: rusqlite::Result<(Fact, String)>| -> Option<Fact> {
                let (mut fact, raw) = r.ok()?;
                fact.object = pool.try_decrypt_content(&raw)?;
                Some(fact)
            };

            let limit_clause = match limit {
                Some(n) => format!(" LIMIT {n} OFFSET {offset}"),
                None => String::new(),
            };
            let facts: Vec<Fact> = if let Some(ns) = namespace {
                let sql = format!(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE superseded_by IS NULL AND (namespace = ?1 OR namespace LIKE ?2)
                       AND id NOT IN (SELECT row_id FROM memory_quarantine WHERE kind = 'fact')
                     ORDER BY rowid DESC{limit_clause}"
                );
                let mut stmt = conn.prepare(&sql)?;
                let prefix = format!("{ns}/%");
                let rows: Vec<Fact> = stmt
                    .query_map(rusqlite::params![ns, &prefix], row_to_raw_fact)?
                    .filter_map(decrypt_filter)
                    .collect();
                rows
            } else {
                let sql = format!(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE superseded_by IS NULL
                       AND id NOT IN (SELECT row_id FROM memory_quarantine WHERE kind = 'fact')
                     ORDER BY rowid DESC{limit_clause}"
                );
                let mut stmt = conn.prepare(&sql)?;
                let rows: Vec<Fact> = stmt
                    .query_map([], row_to_raw_fact)?
                    .filter_map(decrypt_filter)
                    .collect();
                rows
            };
            Ok(facts)
        })?)
    }

    /// Cheap existence probe used by `namespace_is_empty` callers (chat /
    /// recall onboarding hints). Avoids loading + decrypting every fact in
    /// the namespace just to check `.is_empty()`.
    pub fn has_facts_in_namespace(&self, namespace: Option<&str>) -> Result<bool, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let exists: i64 = if let Some(ns) = namespace {
                let prefix = format!("{ns}/%");
                conn.query_row(
                    "SELECT EXISTS(SELECT 1 FROM semantic_facts
                                   WHERE superseded_by IS NULL
                                     AND (namespace = ?1 OR namespace LIKE ?2)
                                   LIMIT 1)",
                    rusqlite::params![ns, &prefix],
                    |row| row.get(0),
                )?
            } else {
                conn.query_row(
                    "SELECT EXISTS(SELECT 1 FROM semantic_facts
                                   WHERE superseded_by IS NULL
                                   LIMIT 1)",
                    [],
                    |row| row.get(0),
                )?
            };
            Ok(exists != 0)
        })?)
    }

    /// List all namespaces with their fact and episode counts.
    ///
    /// SQLite doesn't have `FULL OUTER JOIN`, so the merge is done with
    /// a `UNION ALL` of two per-table count queries reaggregated by
    /// `SUM`. One prepared statement, one round-trip, no Rust-side
    /// HashMap merge — replaces the previous two-query + merge shape
    /// that O(namespaces × hash) was paying for at every call.
    pub fn list_namespaces(&self) -> Result<Vec<NamespaceStats>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT namespace,
                        SUM(fact_count)    AS fact_count,
                        SUM(episode_count) AS episode_count
                 FROM (
                     SELECT namespace, COUNT(*) AS fact_count, 0 AS episode_count
                     FROM semantic_facts
                     WHERE superseded_by IS NULL
                     GROUP BY namespace
                     UNION ALL
                     SELECT namespace, 0 AS fact_count, COUNT(*) AS episode_count
                     FROM episodes
                     GROUP BY namespace
                 )
                 GROUP BY namespace
                 ORDER BY namespace",
            )?;
            let rows: Vec<NamespaceStats> = stmt
                .query_map([], |row| {
                    Ok(NamespaceStats {
                        namespace: row.get(0)?,
                        fact_count: row.get(1)?,
                        episode_count: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?)
    }

    /// Delete a fact from both SQLite and RuVector.
    pub async fn delete_fact(&self, fact_id: &str) -> Result<(), SemanticError> {
        self.db.with_conn(|conn| {
            conn.execute("DELETE FROM semantic_facts WHERE id = ?1", [fact_id])?;
            Ok(())
        })?;

        let ruv_result = self.ruv.delete("facts_vec", fact_id).await;
        if let Err(e) = ruv_result {
            tracing::warn!(
                "RuVector delete failed for {}, re-syncing on next startup",
                fact_id
            );
            return Err(SemanticError::RuVector(e));
        }

        Ok(())
    }

    /// Delete many facts from both SQLite and RuVector in one round-trip
    /// each. Used by `handle_forget` so an N-result match collapses from
    /// N pool-lock acquisitions to one SQL `DELETE … WHERE id IN (…)`
    /// plus one batched RuVector pass. Returns the number of rows the
    /// SQL DELETE actually removed (may be smaller than `ids.len()` if a
    /// fact was already gone). Per-id RuVector failures are logged but
    /// don't fail the call — the deferred re-sync on next startup
    /// reconciles divergence, same as `delete_fact` already does.
    pub async fn delete_facts_batch(&self, ids: &[&str]) -> Result<usize, SemanticError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let deleted = self.db.with_conn(|conn| {
            let placeholders = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("DELETE FROM semantic_facts WHERE id IN ({placeholders})");
            let params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
            let rows = conn.execute(&sql, params.as_slice())?;
            Ok(rows)
        })?;

        let failures = self
            .ruv
            .delete_batch("facts_vec", ids)
            .await
            .map_err(SemanticError::RuVector)?;
        for (id, e) in failures {
            tracing::warn!(
                fact_id = %id,
                "RuVector delete failed (batch), re-syncing on next startup: {e}"
            );
        }
        Ok(deleted)
    }

    /// Find facts matching a Forget target description.
    ///
    /// Used by the Forget intent. The classifier often hands us the whole user
    /// sentence ("Actually forget what I said about my temporary access token")
    /// rather than a distilled topic, so a single `LIKE %<sentence>%` matched
    /// nothing and the forget silently no-op'd. Instead we reduce the target to
    /// its significant content tokens (`temporary`, `access`, `token`) and match
    /// any fact whose subject/predicate/object contains one of them — keying on
    /// the topic, not the filler. A target with no significant tokens (e.g. a
    /// single short word) falls back to the original whole-string substring.
    pub fn find_facts_matching(
        &self,
        query: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        let tokens = significant_forget_tokens(query);
        // Each entry is an escaped `%…%` LIKE pattern. With significant tokens we
        // match ANY of them; with none we fall back to the whole query.
        let patterns: Vec<String> = if tokens.is_empty() {
            let escaped = query.replace('%', r"\%").replace('_', r"\_");
            vec![format!("%{escaped}%")]
        } else {
            tokens.iter().map(|t| format!("%{t}%")).collect()
        };
        // ?1..?N are the content patterns; each is tested against all three
        // columns. The namespace params (if any) follow at ?N+1, ?N+2.
        let content_clause = (1..=patterns.len())
            .map(|i| {
                format!(
                    "(subject LIKE ?{i} ESCAPE '\\' OR predicate LIKE ?{i} ESCAPE '\\' OR object LIKE ?{i} ESCAPE '\\')"
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");

        let mut params: Vec<String> = patterns.clone();
        let sql = if let Some(ns) = namespace {
            let n1 = patterns.len() + 1;
            let n2 = patterns.len() + 2;
            params.push(ns.to_string());
            params.push(format!("{ns}/%"));
            format!(
                "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                 FROM semantic_facts
                 WHERE superseded_by IS NULL
                   AND (namespace = ?{n1} OR namespace LIKE ?{n2})
                   AND ({content_clause})
                 ORDER BY rowid DESC
                 LIMIT 50"
            )
        } else {
            format!(
                "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                 FROM semantic_facts
                 WHERE superseded_by IS NULL
                   AND ({content_clause})
                 ORDER BY rowid DESC
                 LIMIT 50"
            )
        };

        Ok(self.db.with_conn(|conn| {
            let row_to_raw_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(Fact, String)> {
                let raw_object: String = row.get(5)?;
                Ok((
                    Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: String::new(),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(8)?,
                    },
                    raw_object,
                ))
            };
            let decrypt_filter = |r: rusqlite::Result<(Fact, String)>| -> Option<Fact> {
                let (mut fact, raw) = r.ok()?;
                fact.object = pool.try_decrypt_content(&raw)?;
                Some(fact)
            };

            let mut stmt = conn.prepare(&sql)?;
            let param_refs: Vec<&dyn rusqlite::ToSql> =
                params.iter().map(|p| p as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(param_refs.as_slice(), row_to_raw_fact)?;
            Ok(rows.filter_map(decrypt_filter).collect())
        })?)
    }
}

/// Reduce a Forget target phrase to the significant content tokens used to
/// match stored facts. Drops the filler that natural phrasing wraps around the
/// topic ("forget what I said about my X" → `x`), plus tokens of two chars or
/// fewer, so the match keys on the subject rather than the whole sentence.
/// Returns empty when nothing significant remains (caller falls back to a
/// whole-string substring match).
fn significant_forget_tokens(query: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "actually",
        "forget",
        "what",
        "said",
        "about",
        "the",
        "that",
        "please",
        "you",
        "mentioned",
        "told",
        "remember",
        "remembered",
        "stored",
        "regarding",
        "concerning",
        "info",
        "information",
        "anything",
        "everything",
        "was",
        "were",
        "it",
        "this",
        "these",
        "those",
        "and",
        "delete",
        "remove",
        "erase",
        "drop",
        "longer",
        "want",
        "need",
        "keep",
        "our",
        "know",
        "knew",
        "for",
        "with",
        "your",
        "all",
    ];
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 2 && !STOP.contains(&t.as_str()))
        .collect()
}

/// Statistics for a single namespace.
#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub namespace: String,
    pub fact_count: i64,
    pub episode_count: i64,
}
