import { GtagProvider } from '@/components/gtag-provider';
import ReactDOM from 'react-dom/client';
import { loadConfig } from './load-config';
import { prepareGtag } from './prepare-gtag';
import { Widget, type WidgetInstance } from './Widget';

const script = document.currentScript as HTMLScriptElement;
if (!script) {
  throw new Error('Cannot locate document.currentScript');
}

const src = script.src;
const controlled = script.dataset.controlled === 'true';
const trigger = controlled ? true : document.getElementById('tidb-ai-trigger');
const chatEngine = script.dataset.chatEngine;
const scriptDataMeasurementId = script.dataset.measurementId;

loadConfig().then(async ({ settings, bootstrapStatus, experimentalFeatures }) => {
  const measurementId = scriptDataMeasurementId ?? settings.ga_id;

  const gtagFn = measurementId ? prepareGtag(measurementId) : undefined;
  const div = document.createElement('div');

  div.id = 'tidb-ai-widget';
  div.className = 'tidb-ai-widget';
  document.body.appendChild(div);

  const refFn = (current: WidgetInstance) => {
    Object.defineProperty(window, 'tidbai', {
      writable: false,
      value: current,
    });
    window.dispatchEvent(new CustomEvent('tidbaiinitialized', { detail: current }));
  };

  ReactDOM.createRoot(div).render(
    <GtagProvider configured gtagFn={gtagFn} gtagId={measurementId}>
      <Widget
        ref={refFn}
        src={src}
        container={div}
        trigger={trigger}
        exampleQuestions={settings.custom_js_example_questions}
        buttonLabel={settings.custom_js_button_label}
        buttonIcon={settings.custom_js_button_img_src}
        icon={settings.custom_js_logo_src}
        bootstrapStatus={bootstrapStatus}
        experimentalFeatures={experimentalFeatures}
        chatEngine={chatEngine}
      />
    </GtagProvider>,
  );
}).catch((error) => {
  console.error('Failed to initialize tidbai', error);
  Object.defineProperty(window, 'tidbai', {
    writable: false,
    value: {
      open: false,
      dark: false,
      initialized: false,
      error,
    },
  });
  window.dispatchEvent(new CustomEvent('tidbaierror', { detail: error }));
});
