use valence::{
    ai::FFIChatCompletionModel,
    graph::{VNode, VNodeCtx, VTransform},
    Frame, VGraph, VNodeRef,
};

use anyhow::Result;
use valence_runtime_ffi::protos::{BatchChatCompletionRequest, ChatCompletionRequest};

pub struct BatchChatCompletionXform {
    pub model: Box<dyn FFIChatCompletionModel + Send + Sync + 'static>,
}

impl BatchChatCompletionXform {
    pub fn new<M: FFIChatCompletionModel + Send + Sync + 'static>(model: M) -> Self {
        Self {
            model: Box::new(model),
        }
    }
}

impl VNode for BatchChatCompletionXform {
    fn tick(&mut self, ctx: &mut VNodeCtx) -> Result<()> {
        if let Some(Frame::Data(prompts)) = self.recv(ctx) {
            let requests = prompts
                .into_iter()
                .map(|prompt| ChatCompletionRequest {
                    prompt,
                    model: self.model.ffi_model() as i32,
                })
                .collect::<Vec<_>>();

            let request = BatchChatCompletionRequest { requests };
            let messages: Vec<String> = valence::ai::batch_chat_completion_request(request)?;

            self.send(ctx, Frame::Data(messages))?;
        }

        if ctx.recv_finished() {
            self.send(ctx, Frame::End)?;
        }

        Ok(())
    }

    fn default_label(&self) -> Option<String> {
        Some("BatchChatCompletion".to_owned())
    }
}

impl VTransform for BatchChatCompletionXform {
    type Input = Vec<String>;
    type Output = Vec<String>;
}

pub trait ChatCompletionNodeOps {
    fn batch_chat_completion<M: FFIChatCompletionModel + Send + Sync + 'static>(
        &self,
        g: &mut VGraph,
        model: M,
    ) -> VNodeRef<Vec<String>, Vec<String>>;
}

impl<I> ChatCompletionNodeOps for VNodeRef<I, Vec<String>> {
    fn batch_chat_completion<M: FFIChatCompletionModel + Send + Sync + 'static>(
        &self,
        g: &mut VGraph,
        model: M,
    ) -> VNodeRef<Vec<String>, Vec<String>> {
        self.transform(g, BatchChatCompletionXform::new(model))
    }
}
