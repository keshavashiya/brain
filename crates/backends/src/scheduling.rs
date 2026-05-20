//! Scheduling backend — persists scheduled intents to SQLite.

#[derive(Clone)]
pub struct DefaultSchedulingBackend {
    pub db: storage::SqlitePool,
    pub mode: brain::config::SchedulingMode,
}

#[async_trait::async_trait]
impl cortex::actions::SchedulingBackend for DefaultSchedulingBackend {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<cortex::actions::ScheduleOutcome, cortex::actions::ActionError> {
        if self.mode != brain::config::SchedulingMode::PersistOnly {
            return Err(cortex::actions::ActionError::InvalidArguments(format!(
                "Unsupported scheduling mode: {:?}",
                self.mode
            )));
        }

        let metadata = serde_json::json!({
            "source": "action_dispatcher",
            "mode": "persist_only",
        })
        .to_string();

        let schedule_id = self
            .db
            .insert_scheduled_intent(description, cron, namespace, Some(&metadata))
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(cortex::actions::ScheduleOutcome {
            schedule_id,
            status: "scheduled".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduling_backend_persists_intent() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let backend = DefaultSchedulingBackend {
            db: db.clone(),
            mode: brain::config::SchedulingMode::PersistOnly,
        };

        let outcome = cortex::actions::SchedulingBackend::schedule(
            &backend,
            "ship release",
            Some("0 9 * * 1-5"),
            "work",
        )
        .await
        .unwrap();

        assert_eq!(outcome.status, "scheduled");
        let intents = db.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].id, outcome.schedule_id);
        assert_eq!(intents[0].description, "ship release");
    }
}
