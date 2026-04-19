use std::sync::Arc;

use cortex::actions::Action;

use crate::{Classification, Intent, IntentClassifier, IntentFallback, NormalizedMessage};

/// Routes normalized messages to appropriate handlers.
pub struct SignalRouter {
    classifier: IntentClassifier,
}

impl SignalRouter {
    pub fn new() -> Self {
        Self {
            classifier: IntentClassifier::new(),
        }
    }

    pub fn with_llm_fallback(mut self, fallback: Arc<dyn IntentFallback>) -> Self {
        self.classifier = self.classifier.with_llm_fallback(fallback);
        self
    }

    pub async fn route(&self, message: &NormalizedMessage) -> Classification {
        self.classifier.classify(&message.content).await
    }

    /// Returns `None` for intents handled directly in SignalProcessor.
    pub fn intent_to_action(&self, intent: &Intent) -> Option<Action> {
        match intent {
            Intent::StoreFact {
                subject,
                predicate,
                object,
            } => Some(Action::StoreFact {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            }),
            Intent::Recall { query } => Some(Action::Recall {
                query: query.clone(),
            }),
            Intent::ExecuteCommand { command, args } => Some(Action::ExecuteCommand {
                command: command.clone(),
                args: args.clone(),
            }),
            Intent::WebSearch { query } => Some(Action::WebSearch {
                query: query.clone(),
            }),
            Intent::Schedule { description, cron } => Some(Action::ScheduleTask {
                description: description.clone(),
                cron: cron.clone(),
            }),
            Intent::SendMessage {
                channel,
                recipient,
                content,
            } => Some(Action::SendMessage {
                channel: channel.clone(),
                recipient: recipient.clone(),
                content: content.clone(),
            }),
            _ => None,
        }
    }
}

impl Default for SignalRouter {
    fn default() -> Self {
        Self::new()
    }
}
