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

    /// List all active facts in a specific namespace.
    pub fn list_by_namespace(&self, namespace: Option<&str>) -> Result<Vec<Fact>, SemanticError> {
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
                     WHERE superseded_by IS NULL AND (namespace = ?1 OR namespace LIKE ?2)
                     ORDER BY rowid DESC",
                )?;
                let prefix = format!("{ns}/%");
                let result = stmt
                    .query_map(rusqlite::params![ns, &prefix], row_to_raw_fact)?
                    .filter_map(decrypt_filter)
                    .collect();
                result
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE superseded_by IS NULL
                     ORDER BY rowid DESC",
                )?;
                let result = stmt
                    .query_map([], row_to_raw_fact)?
                    .filter_map(decrypt_filter)
                    .collect();
                result
            };
            Ok(facts)
        })?)
    }

    /// Cheap existence probe used by `namespace_is_empty` callers (chat /
    /// recall onboarding hints). Avoids loading + decrypting every fact in
    /// the namespace just to check `.is_empty()`.
    pub fn has_facts_in_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<bool, SemanticError> {
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
    pub fn list_namespaces(&self) -> Result<Vec<NamespaceStats>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT namespace, COUNT(*) as fact_count FROM semantic_facts
                 WHERE superseded_by IS NULL
                 GROUP BY namespace ORDER BY namespace",
            )?;
            let fact_ns: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            let mut stmt2 = conn.prepare(
                "SELECT namespace, COUNT(*) as ep_count FROM episodes
                 GROUP BY namespace ORDER BY namespace",
            )?;
            let ep_ns: Vec<(String, i64)> = stmt2
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            let mut map: std::collections::HashMap<String, (i64, i64)> =
                std::collections::HashMap::new();
            for (ns, cnt) in &fact_ns {
                map.entry(ns.clone()).or_default().0 = *cnt;
            }
            for (ns, cnt) in &ep_ns {
                map.entry(ns.clone()).or_default().1 = *cnt;
            }

            let mut result: Vec<NamespaceStats> = map
                .into_iter()
                .map(|(namespace, (fact_count, episode_count))| NamespaceStats {
                    namespace,
                    fact_count,
                    episode_count,
                })
                .collect();
            result.sort_by(|a, b| a.namespace.cmp(&b.namespace));
            Ok(result)
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

    /// Find facts whose subject, predicate, or object contains the query string.
    ///
    /// Used by the Forget intent to find facts matching a target description.
    pub fn find_facts_matching(
        &self,
        query: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        let escaped = query.replace('%', r"\%").replace('_', r"\_");
        let pattern = format!("%{escaped}%");
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
                     WHERE superseded_by IS NULL
                       AND (namespace = ?2 OR namespace LIKE ?3)
                       AND (subject LIKE ?1 ESCAPE '\\' OR predicate LIKE ?1 ESCAPE '\\' OR object LIKE ?1 ESCAPE '\\')
                     ORDER BY rowid DESC
                     LIMIT 50",
                )?;
                let prefix = format!("{ns}/%");
                let rows =
                    stmt.query_map(rusqlite::params![&pattern, ns, &prefix], row_to_raw_fact)?;
                rows.filter_map(decrypt_filter).collect()
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE superseded_by IS NULL
                       AND (subject LIKE ?1 ESCAPE '\\' OR predicate LIKE ?1 ESCAPE '\\' OR object LIKE ?1 ESCAPE '\\')
                     ORDER BY rowid DESC
                     LIMIT 50",
                )?;
                let rows = stmt.query_map([&pattern], row_to_raw_fact)?;
                rows.filter_map(decrypt_filter).collect()
            };

            Ok(facts)
        })?)
    }
}

/// Statistics for a single namespace.
#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub namespace: String,
    pub fact_count: i64,
    pub episode_count: i64,
}
