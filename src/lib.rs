use mixlayer::{
    ai::FFIChatCompletionModel,
    graph::{MxlNode, MxlNodeCtx, MxlTransform},
    Frame, MxlGraph, MxlNodeRef,
};

use anyhow::Result;
use mixlayer_runtime_ffi::protos::{BatchChatCompletionRequest, ChatCompletionRequest};

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

impl MxlNode for BatchChatCompletionXform {
    fn tick(&mut self, ctx: &mut MxlNodeCtx) -> Result<()> {
        if let Some(Frame::Data(prompts)) = self.recv(ctx) {
            let requests = prompts
                .into_iter()
                .map(|prompt| ChatCompletionRequest {
                    prompt,
                    model: self.model.ffi_model() as i32,
                })
                .collect::<Vec<_>>();

            let request = BatchChatCompletionRequest { requests };
            let messages: Vec<String> = mixlayer::ai::batch_chat_completion_request(request)?;

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

impl MxlTransform for BatchChatCompletionXform {
    type Input = Vec<String>;
    type Output = Vec<String>;
}

pub trait ChatCompletionNodeOps {
    fn batch_chat_completion<M: FFIChatCompletionModel + Send + Sync + 'static>(
        &self,
        g: &mut MxlGraph,
        model: M,
    ) -> MxlNodeRef<Vec<String>, Vec<String>>;
}

impl<I> ChatCompletionNodeOps for MxlNodeRef<I, Vec<String>> {
    fn batch_chat_completion<M: FFIChatCompletionModel + Send + Sync + 'static>(
        &self,
        g: &mut MxlGraph,
        model: M,
    ) -> MxlNodeRef<Vec<String>, Vec<String>> {
        self.transform(g, BatchChatCompletionXform::new(model))
    }
}
