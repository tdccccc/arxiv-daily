import {
  PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS,
  PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_NAME_LENGTH,
  PERSONAL_LIBRARY_MAX_ID_LENGTH,
  PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES,
  PERSONAL_LIBRARY_MAX_PROPOSAL_TOPICS,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
  PERSONAL_LIBRARY_MIN_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MIN_REPRESENTATIVES,
} from "./personal-library-interest-profile";

export const PERSONAL_LIBRARY_ORGANIZATION_MIN_TOPICS = 2 as const;
export const PERSONAL_LIBRARY_ORGANIZATION_MAX_TOPICS = 4 as const;
export const PERSONAL_LIBRARY_ORGANIZATION_MAX_DIRECTIONS_PER_TOPIC = 2 as const;

export interface PersonalLibraryOrganizationGroup {
  id: string;
  papers: readonly { paperKey: string }[];
}

/** The existing settings needed for organization, without host-only fields. */
export interface PersonalLibraryExistingTopic {
  readonly id: string;
  readonly name: string;
  readonly directions: readonly { readonly id: string; readonly text: string }[];
}

export interface PersonalLibraryCoveredGroup {
  groupId: string;
  topicId: string;
  directionId: string;
}

export interface OrganizedDirection {
  text: string;
  discoveryCues: string[];
  groupIds: string[];
  representativePaperKeys: string[];
}

export interface OrganizedTopic {
  suggestedName: string;
  targetTopicId?: string;
  directions: OrganizedDirection[];
}

export interface OrganizedTopicsResult {
  topics: OrganizedTopic[];
  coveredGroups?: PersonalLibraryCoveredGroup[];
}

export type OrganizationValidationReason =
  | "not-json"
  | "wrong-shape"
  | "topic-count"
  | "direction-count"
  | "group-assignment"
  | "text-bounds"
  | "cues-invalid"
  | "representatives-invalid"
  | "reference-out-of-scope"
  | "member-count";

/** Validate a complete partition of evidence groups before constructing proposals. */
export function decodeOrganizedTopics(
  raw: string,
  groups: readonly PersonalLibraryOrganizationGroup[],
  existingTopics: readonly PersonalLibraryExistingTopic[] = [],
): { ok: true; value: OrganizedTopicsResult }
  | { ok: false; reason: OrganizationValidationReason } {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { ok: false, reason: "not-json" };
  }
  if (!isExactObject(value, ["topics"], ["coveredGroups"]) || !Array.isArray(value.topics)) {
    return { ok: false, reason: "wrong-shape" };
  }
  const rawCoveredGroups = Object.hasOwn(value, "coveredGroups") ? value.coveredGroups : [];
  if (!Array.isArray(rawCoveredGroups)) return { ok: false, reason: "wrong-shape" };
  const minimumTopics = existingTopics.length > 0 ? 0
    : groups.length === 1 ? 1 : PERSONAL_LIBRARY_ORGANIZATION_MIN_TOPICS;
  const maximumTopics = existingTopics.length > 0 ? PERSONAL_LIBRARY_MAX_PROPOSAL_TOPICS
    : PERSONAL_LIBRARY_ORGANIZATION_MAX_TOPICS;
  if (value.topics.length < minimumTopics
    || value.topics.length > Math.min(maximumTopics, groups.length)) {
    return { ok: false, reason: "topic-count" };
  }

  const groupById = new Map<string, PersonalLibraryOrganizationGroup>();
  for (const group of groups) {
    if (groupById.has(group.id) || group.papers.length === 0) {
      return { ok: false, reason: "group-assignment" };
    }
    groupById.set(group.id, group);
  }
  const existingById = new Map<string, PersonalLibraryExistingTopic>();
  for (const topic of existingTopics) {
    if (!isOpaqueId(topic.id) || existingById.has(topic.id)) {
      return { ok: false, reason: "reference-out-of-scope" };
    }
    existingById.set(topic.id, topic);
  }
  const assignedGroups = new Set<string>();
  const coveredGroups: PersonalLibraryCoveredGroup[] = [];
  for (const rawCoverage of rawCoveredGroups) {
    if (!isExactObject(rawCoverage, ["groupId", "topicId", "directionId"])) {
      return { ok: false, reason: "wrong-shape" };
    }
    if (typeof rawCoverage.groupId !== "string"
      || !groupById.has(rawCoverage.groupId) || assignedGroups.has(rawCoverage.groupId)) {
      return { ok: false, reason: "group-assignment" };
    }
    if (!isOpaqueId(rawCoverage.topicId) || !isOpaqueId(rawCoverage.directionId)) {
      return { ok: false, reason: "reference-out-of-scope" };
    }
    const topic = existingById.get(rawCoverage.topicId);
    const matchingDirections = topic?.directions.filter(({ id }) => id === rawCoverage.directionId) ?? [];
    if (matchingDirections.length !== 1 || !matchingDirections[0]!.text.trim()) {
      return { ok: false, reason: "reference-out-of-scope" };
    }
    assignedGroups.add(rawCoverage.groupId);
    coveredGroups.push({ groupId: rawCoverage.groupId, topicId: rawCoverage.topicId, directionId: rawCoverage.directionId });
  }
  const topics: OrganizedTopic[] = [];
  const extendedTopics = new Set<string>();
  let newTopics = 0;
  let directionCount = 0;
  for (const rawTopic of value.topics) {
    if (!isExactObject(rawTopic, ["suggestedName", "directions"], ["targetTopicId"]) || !Array.isArray(rawTopic.directions)) {
      return { ok: false, reason: "wrong-shape" };
    }
    if (!isSingleLineText(rawTopic.suggestedName, PERSONAL_LIBRARY_MAX_NAME_LENGTH)) {
      return { ok: false, reason: "text-bounds" };
    }
    let target: PersonalLibraryExistingTopic | undefined;
    if (Object.hasOwn(rawTopic, "targetTopicId")) {
      if (!isOpaqueId(rawTopic.targetTopicId)
        || !(target = existingById.get(rawTopic.targetTopicId))
        || extendedTopics.has(rawTopic.targetTopicId)) {
        return { ok: false, reason: "reference-out-of-scope" };
      }
      extendedTopics.add(target.id);
    } else {
      if (existingTopics.some(({ name }) => name === rawTopic.suggestedName)) {
        return { ok: false, reason: "reference-out-of-scope" };
      }
      newTopics += 1;
      if (newTopics > PERSONAL_LIBRARY_ORGANIZATION_MAX_TOPICS) {
        return { ok: false, reason: "topic-count" };
      }
    }
    directionCount += rawTopic.directions.length;
    if (rawTopic.directions.length < 1
      || rawTopic.directions.length > PERSONAL_LIBRARY_ORGANIZATION_MAX_DIRECTIONS_PER_TOPIC
      || directionCount > PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES) {
      return { ok: false, reason: "direction-count" };
    }
    const directions: OrganizedDirection[] = [];
    for (const rawDirection of rawTopic.directions) {
      if (!isExactObject(rawDirection, ["text", "discoveryCues", "groupIds", "representativePaperKeys"])) {
        return { ok: false, reason: "wrong-shape" };
      }
      if (!isSingleLineText(rawDirection.text, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH)) {
        return { ok: false, reason: "text-bounds" };
      }
      if (!isUniqueStringArray(rawDirection.discoveryCues)
        || rawDirection.discoveryCues.length < PERSONAL_LIBRARY_MIN_DISCOVERY_CUES
        || rawDirection.discoveryCues.length > PERSONAL_LIBRARY_MAX_DISCOVERY_CUES
        || !rawDirection.discoveryCues.every((cue) => isBoundedText(cue, PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH))) {
        return { ok: false, reason: "cues-invalid" };
      }
      if (!isUniqueStringArray(rawDirection.groupIds) || rawDirection.groupIds.length === 0) {
        return { ok: false, reason: "group-assignment" };
      }

      // Representatives are evidence for this direction, so their scope is
      // its assigned groups' full member union, not the whole library.
      const members = new Set<string>();
      for (const groupId of rawDirection.groupIds) {
        const group = groupById.get(groupId);
        if (!group || assignedGroups.has(groupId)) {
          return { ok: false, reason: "group-assignment" };
        }
        assignedGroups.add(groupId);
        for (const paper of group.papers) members.add(paper.paperKey);
      }
      if (members.size > PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS) {
        return { ok: false, reason: "member-count" };
      }
      if (!isUniqueStringArray(rawDirection.representativePaperKeys)
        || rawDirection.representativePaperKeys.length < PERSONAL_LIBRARY_MIN_REPRESENTATIVES
        || rawDirection.representativePaperKeys.length > PERSONAL_LIBRARY_MAX_REPRESENTATIVES) {
        return { ok: false, reason: "representatives-invalid" };
      }
      if (!rawDirection.representativePaperKeys.every((key) => members.has(key))) {
        return { ok: false, reason: "reference-out-of-scope" };
      }
      directions.push({
        text: rawDirection.text,
        discoveryCues: [...rawDirection.discoveryCues].sort(codeUnitCompare),
        groupIds: [...rawDirection.groupIds].sort(codeUnitCompare),
        representativePaperKeys: [...rawDirection.representativePaperKeys].sort(codeUnitCompare),
      });
    }
    topics.push({
      suggestedName: target && isSingleLineText(target.name, PERSONAL_LIBRARY_MAX_NAME_LENGTH)
        ? target.name : rawTopic.suggestedName,
      ...(target ? { targetTopicId: target.id } : {}),
      directions,
    });
  }
  if (assignedGroups.size !== groupById.size) {
    return { ok: false, reason: "group-assignment" };
  }
  coveredGroups.sort((left, right) => codeUnitCompare(left.groupId, right.groupId));
  return { ok: true, value: { topics, coveredGroups } };
}

function isExactObject(value: unknown, keys: readonly string[], optionalKeys: readonly string[] = []): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.keys(value).every((key) => keys.includes(key) || optionalKeys.includes(key))
    && keys.every((key) => Object.hasOwn(value, key));
}

function isOpaqueId(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= PERSONAL_LIBRARY_MAX_ID_LENGTH
    && /^[A-Za-z0-9._~-]+$/.test(value);
}

function isBoundedText(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum
    && value.trim() === value;
}

function isSingleLineText(value: unknown, maximum: number): value is string {
  return isBoundedText(value, maximum) && !/[\r\n\u2028\u2029]/u.test(value);
}

function isUniqueStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    && new Set(value).size === value.length;
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
