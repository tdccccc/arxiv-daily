import type {
  PersonalLibraryClusterMember,
  PersonalLibraryRepresentativeEvidence,
} from "../personal-library-interest-profile";

/**
 * Everything the incremental scoring stages read off a direction a new
 * candidate might be placed into: an identity to report, a display name, the
 * papers already behind it, and whether the researcher locked it.
 *
 * This used to be `PersonalLibraryConfirmedDirection`, a record of the
 * confirmed interest profile document. That document retired with ADR 0012, so
 * the type is stated here as what the scoring needs rather than borrowed from
 * a store that no longer exists. Scoring never cared where directions live —
 * P5 points these stages at `settings.topics` (ADR 0014 §2/§3), and only the
 * adapter that builds this shape has to change.
 */
export interface PlaceableDirection {
  id: string;
  name: string;
  representatives: PersonalLibraryRepresentativeEvidence[];
  clusterMembers: PersonalLibraryClusterMember[];
  /** Present when the researcher locked the direction against automatic changes. */
  lockedAt?: string;
}
