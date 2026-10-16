#!/usr/bin/env node
/** Generate multilingual, held-out candidate evaluation sets for Talmur NER.
 *
 * The outputs are deliberately labelled candidate sets: they are synthetic,
 * although they use templates and values separate from the training corpus.
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const outDir = path.join(root, "data", "gold");
const count = 20;

const canon = {
  NOME: "PERSON_FIRSTNAME", COGNOME: "PERSON_LASTNAME", EMAIL: "EMAIL_ADDRESS",
  NUMERO_TELEFONO: "PHONE_NUMBER", IBAN: "IBAN_CODE", BIC: "SWIFT_CODE",
  CARTELLA_CLINICA: "MEDICAL_RECORD", MALATTIA: "DISEASE", MEDICINA: "MEDICATION",
  IP: "IP_ADDRESS", MAC: "MAC_ADDRESS", PASSWORD: "PASSWORD", IMPORTO: "AMOUNT",
};

const languages = {
  it: { first: ["Elisa", "Marco", "Chiara", "Davide"], last: ["Fontana", "Serra", "Riva", "Gallo"],
    sensitive: [
      (p) => `Richiesta di supporto ${p.n}: ${p.first} ${p.last} chiede risposta a ${p.email} oppure al ${p.phone}.`,
      (p) => `Per il pagamento ${p.n} verificare l'IBAN ${p.iban} e il codice BIC ${p.bic}.`,
      (p) => `Nel fascicolo ${p.record} è riportata ${p.disease}; terapia indicata: ${p.med}.`,
      (p) => `L'accesso anomalo ${p.n} proviene dall'IP ${p.ip}, MAC ${p.mac}; password temporanea ${p.password}.`,
    ],
    benign: (p) => `Nella guida pubblica ${p.first} ${p.last} è un personaggio fittizio; l'esempio usa un importo di ${p.amount} euro.`,
    negative: (n) => `La nota informativa numero ${n} descrive le funzionalità disponibili nella prossima versione del servizio.`, },
  en: { first: ["Maya", "Ethan", "Olivia", "Daniel"], last: ["Harper", "Reed", "Parker", "Morgan"],
    sensitive: [
      (p) => `Support request ${p.n}: ${p.first} ${p.last} asks for a reply at ${p.email} or ${p.phone}.`,
      (p) => `For payment reference ${p.n}, verify IBAN ${p.iban} and BIC ${p.bic}.`,
      (p) => `File ${p.record} notes ${p.disease}; the listed medication is ${p.med}.`,
      (p) => `Security alert ${p.n} came from IP ${p.ip}, MAC ${p.mac}; temporary password ${p.password}.`,
    ],
    benign: (p) => `In the public tutorial, ${p.first} ${p.last} is a fictional example and the sample amount is ${p.amount} euros.`,
    negative: (n) => `Release note ${n} outlines the improvements planned for the next version of the service.`, },
  fr: { first: ["Lucie", "Hugo", "Camille", "Nicolas"], last: ["Morel", "Lefevre", "Renaud", "Blanc"],
    sensitive: [
      (p) => `Demande d'assistance ${p.n} : ${p.first} ${p.last} souhaite une réponse à ${p.email} ou au ${p.phone}.`,
      (p) => `Pour le paiement ${p.n}, vérifier l'IBAN ${p.iban} et le BIC ${p.bic}.`,
      (p) => `Le dossier ${p.record} mentionne ${p.disease} ; le médicament indiqué est ${p.med}.`,
      (p) => `Alerte de sécurité ${p.n} depuis l'IP ${p.ip}, MAC ${p.mac} ; mot de passe temporaire ${p.password}.`,
    ],
    benign: (p) => `Dans le guide public, ${p.first} ${p.last} est un personnage fictif et l'exemple utilise ${p.amount} euros.`,
    negative: (n) => `La note de version ${n} présente les améliorations prévues pour le prochain service.`, },
  de: { first: ["Lea", "Jonas", "Mira", "Felix"], last: ["Koch", "Neumann", "Wolf", "Brandt"],
    sensitive: [
      (p) => `Supportanfrage ${p.n}: ${p.first} ${p.last} bittet um Antwort an ${p.email} oder ${p.phone}.`,
      (p) => `Für Zahlung ${p.n} bitte IBAN ${p.iban} und BIC ${p.bic} prüfen.`,
      (p) => `In Akte ${p.record} steht ${p.disease}; als Medikament ist ${p.med} angegeben.`,
      (p) => `Sicherheitsmeldung ${p.n} von IP ${p.ip}, MAC ${p.mac}; temporäres Passwort ${p.password}.`,
    ],
    benign: (p) => `Im öffentlichen Leitfaden ist ${p.first} ${p.last} eine fiktive Person; das Beispiel nutzt ${p.amount} Euro.`,
    negative: (n) => `Die Versionsnotiz ${n} erläutert die geplanten Verbesserungen des nächsten Dienstes.`, },
  es: { first: ["Irene", "Mateo", "Clara", "Bruno"], last: ["Santos", "Vega", "Prieto", "Molina"],
    sensitive: [
      (p) => `Solicitud de soporte ${p.n}: ${p.first} ${p.last} pide respuesta en ${p.email} o en ${p.phone}.`,
      (p) => `Para el pago ${p.n}, verifique el IBAN ${p.iban} y el BIC ${p.bic}.`,
      (p) => `El expediente ${p.record} indica ${p.disease}; el medicamento registrado es ${p.med}.`,
      (p) => `Alerta de seguridad ${p.n} desde la IP ${p.ip}, MAC ${p.mac}; contraseña temporal ${p.password}.`,
    ],
    benign: (p) => `En la guía pública, ${p.first} ${p.last} es un personaje ficticio y el ejemplo usa ${p.amount} euros.`,
    negative: (n) => `La nota de versión ${n} explica las mejoras previstas para el próximo servicio.`, },
  pt: { first: ["Ines", "Tiago", "Rita", "Caio"], last: ["Mendes", "Rocha", "Pires", "Faria"],
    sensitive: [
      (p) => `Pedido de apoio ${p.n}: ${p.first} ${p.last} pede resposta para ${p.email} ou ${p.phone}.`,
      (p) => `Para o pagamento ${p.n}, confirme o IBAN ${p.iban} e o BIC ${p.bic}.`,
      (p) => `O processo ${p.record} indica ${p.disease}; o medicamento registado é ${p.med}.`,
      (p) => `Alerta de segurança ${p.n} do IP ${p.ip}, MAC ${p.mac}; palavra-passe temporária ${p.password}.`,
    ],
    benign: (p) => `No guia público, ${p.first} ${p.last} é uma personagem fictícia e o exemplo usa ${p.amount} euros.`,
    negative: (n) => `A nota de versão ${n} explica as melhorias previstas para o próximo serviço.`, },
};

const entity = (surface, label, sensitivity, action, context_tags) => ({ surface, label, canonical_entity: canon[label], sensitivity, action, context_tags });
const props = (language, n) => ({
  n, first: languages[language].first[n % 4], last: languages[language].last[(n + 1) % 4],
  email: `contact${n}.${language}@example.test`, phone: `+39 347 55${String(n).padStart(4, "0")}`,
  iban: `IT60X0542811101000000${String(120000 + n).slice(-6)}`, bic: `TSTIIT${String(n).padStart(2, "0")}`,
  record: `MR-${language.toUpperCase()}-${String(1000 + n)}`, disease: ["asma", "ipertensione", "diabete", "emicrania"][n % 4],
  med: ["Metformina", "Ibuprofene", "Amoxicillina", "Atorvastatina"][n % 4], ip: `10.42.${Math.floor(n / 10)}.${n + 10}`,
  mac: `02:42:AC:11:${String(n).padStart(2, "0")}:0A`, password: `Tmp-${language}-${2026 + n}!`, amount: `${100 + n},50`,
});

function sensitiveEntities(p, variant) {
  if (variant === 0) return [entity(p.first, "NOME", "high", "mask", ["support"]), entity(p.last, "COGNOME", "high", "mask", ["support"]), entity(p.email, "EMAIL", "high", "mask", ["contact"]), entity(p.phone, "NUMERO_TELEFONO", "high", "mask", ["contact"])];
  if (variant === 1) return [entity(p.iban, "IBAN", "critical", "mask", ["payment"]), entity(p.bic, "BIC", "high", "mask", ["payment"])];
  if (variant === 2) return [entity(p.record, "CARTELLA_CLINICA", "critical", "mask", ["health"]), entity(p.disease, "MALATTIA", "critical", "mask", ["health"]), entity(p.med, "MEDICINA", "high", "mask", ["health"])];
  return [entity(p.ip, "IP", "medium", "mask", ["security"]), entity(p.mac, "MAC", "medium", "mask", ["security"]), entity(p.password, "PASSWORD", "critical", "mask", ["secret"])];
}

for (const language of Object.keys(languages)) {
  const rows = [];
  for (let i = 1; i <= count; i += 1) {
    const p = props(language, i); const variant = i % 4;
    rows.push({ id: `gold-${language}-sensitive-${String(i).padStart(3, "0")}`, language, split: "test", source: { kind: "synthetic_generated", template_id: `candidate-sensitive-v2-${variant}`, document_type: "held_out_candidate", notes: "Synthetic candidate; review required for production benchmark." }, text: languages[language].sensitive[variant](p), entities: sensitiveEntities(p, variant), record_tags: ["gold_candidate", "sensitive", language] });
  }
  for (let i = 1; i <= count; i += 1) {
    const p = props(language, i + count);
    rows.push({ id: `gold-${language}-benign-${String(i).padStart(3, "0")}`, language, split: "test", source: { kind: "synthetic_generated", template_id: "candidate-benign-v2", document_type: "held_out_candidate", notes: "Synthetic candidate; review required for production benchmark." }, text: languages[language].benign(p), entities: [entity(p.first, "NOME", "low", "keep", ["fictional", "documentation"]), entity(p.last, "COGNOME", "low", "keep", ["fictional", "documentation"]), entity(p.amount, "IMPORTO", "low", "keep", ["public_example"])], record_tags: ["gold_candidate", "benign_context", language] });
  }
  for (let i = 1; i <= count; i += 1) {
    rows.push({ id: `gold-${language}-negative-${String(i).padStart(3, "0")}`, language, split: "test", source: { kind: "synthetic_generated", template_id: "candidate-negative-v2", document_type: "held_out_candidate", notes: "Synthetic candidate; review required for production benchmark." }, text: languages[language].negative(i), entities: [], record_tags: ["gold_candidate", "negative", language] });
  }
  const output = path.join(outDir, `${language}.gold.candidate.v1.jsonl`);
  fs.writeFileSync(output, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
  console.log(`${path.basename(output)}: ${rows.length} records (20 sensitive, 20 benign_context, 20 negative)`);
}
