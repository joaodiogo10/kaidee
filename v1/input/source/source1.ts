"use client";

import { TriangleAlert } from "lucide-react";

import usePermissionCheck from "@/features/auth/hooks/usePermissionCheck";
import { useProjectDeleteAction } from "@/features/project/projectActions";
import { useProjectQuery } from "@/features/project/projectQueries";
import PageSection from "@/lib/components/ui/sections/PageSection";
import { RolePermission } from "@/shared/organization/role/types";
import DangerZone, { DangerActionRow } from "@/shared/ui/components/DangerZone";

interface ProjectDangerZoneProps {
  id: string;
}

const ProjectDangerZone: React.FC<ProjectDangerZoneProps> = ({ id }) => {
  const {
    isRefetching: projectRefreshing,
    error: projectError,
    refetch: projectRefetch,
  } = useProjectQuery();

  const deleteApplication = useProjectDeleteAction();
  const hasDeleteApplicationPermission = usePermissionCheck(RolePermission["application:delete"]);

  return (
    <PageSection
      id={id}
      icon={TriangleAlert}
      title="Project Danger Zone"
      description="These actions are irreversible. Please be certain before proceeding!"
      viewPermission={RolePermission["application:settings:view"]}
      error={projectError !== null}
      onRefetch={projectRefetch}
      isFetching={projectRefreshing}
    >
      {hasDeleteApplicationPermission && (
        <div className="mt-2">
          <DangerZone>
            <DangerActionRow
              title="Delete Project"
              description="This will permanently delete the project and all associated data. This action cannot be undone."
              actionLabel="Delete Project"
              onAction={deleteApplication.execute}
              isLoading={deleteApplication.isPending}
              permission={hasDeleteApplicationPermission}
              tooltipLabel="You don't have permission to delete this project"
            />
          </DangerZone>
        </div>
      )}
    </PageSection>
  );
};

export default ProjectDangerZone;
